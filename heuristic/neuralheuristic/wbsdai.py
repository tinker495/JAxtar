from functools import partial
from typing import Callable

import chex
import jax
import jax.numpy as jnp

from heuristic.neuralheuristic.neuralheuristic_base import (
    NeuralHeuristicBase as NeuralHeuristic,
)
from JAxtar.astar import astar_builder
from JAxtar.search_base import SearchResult
from puzzle.puzzle_base import Puzzle


def get_one_solved_branch_samples(
    puzzle: Puzzle,
    heuristic: NeuralHeuristic,
    astar_fn: Callable[[Puzzle.SolveConfig, Puzzle.State, jax.tree_util.PyTreeDef], SearchResult],
    max_depth: int,
    topk_branch_size: int,
    topk_branch_ratio: float,
    heuristic_params: jax.tree_util.PyTreeDef,
    key: chex.PRNGKey,
):
    solve_config, initial_state = puzzle.get_inits(key)

    search_result = astar_fn(solve_config, initial_state, heuristic_params)

    leafs, paths, masks = search_result.get_top_k_branchs_paths(topk_branch_size, max_depth - 1)
    # leafs: [topk_branch_size, ...]
    # paths: [topk_branch_size, max_depth - 1, ...]
    # masks: [topk_branch_size, max_depth - 1]
    masks = jnp.concatenate((jnp.ones(masks.shape[0], dtype=bool)[:, jnp.newaxis], masks), axis=1)

    leaf_states = search_result.get_state(leafs)
    leaf_costs = search_result.get_cost(leafs)
    # leaf_states: [topk_branch_size, ...], leaf_costs: [topk_branch_size]
    leaf_solve_configs = puzzle.batched_hindsight_transform(leaf_states)  # states -> solve_configs
    # leaf_solve_configs: [topk_branch_size, ...]

    costs_threshold = jnp.max(leaf_costs) * (
        1 - topk_branch_ratio
    )  # only samples topk_branch_ratio of the branchs
    masks = jnp.where((leaf_costs >= costs_threshold)[:, jnp.newaxis], masks, False)

    path_states = search_result.get_state(paths)
    path_states = jax.tree_util.tree_map(
        lambda x, y: jnp.concatenate((x, y), axis=1), leaf_states[:, jnp.newaxis], path_states
    )
    path_costs = search_result.get_cost(paths)
    path_costs = jnp.concatenate((leaf_costs[:, jnp.newaxis], path_costs), axis=1)
    # path_states: [topk_branch_size, max_depth, ...], path_costs: [topk_branch_size, max_depth]

    raw_costs = leaf_costs[:, jnp.newaxis] - path_costs
    raw_costs = jnp.where(masks, raw_costs, leaf_costs[:, jnp.newaxis])
    # raw_costs: [topk_branch_size, max_depth] , [[1, 2, 3, 4, 5, ...], [1, 2, 3, 4, 5, ...], ...]
    # example: [1, 2, 3, 4, 5, 6, 7, ... , 20, 21, 22, 22, 22, 22, 22, ...]

    incr_costs = raw_costs[:, 1:] - raw_costs[:, :-1]
    incr_costs = jnp.concatenate((raw_costs[:, 0, jnp.newaxis], incr_costs), axis=1)
    # incr_costs: [topk_branch_size, max_depth] , [[1, 1, 1, 1, 1, ...], [1, 1, 1, 1, 1, ...], ...]

    is_solved = jax.vmap(jax.vmap(puzzle.is_solved, in_axes=(None, 0)))(
        leaf_solve_configs, path_states
    )
    # is_solved: [topk_branch_size, max_depth]
    incr_costs = jnp.where(is_solved, 0, incr_costs)
    # incr_costs: [topk_branch_size, max_depth] , [[0, 0, 1, 1, 1, ...], [0, 0, 0, 1, 1, ...], ...]

    true_costs = jnp.cumsum(incr_costs, axis=1)
    # true_costs: [topk_branch_size, max_depth] , [[0, 0, 1, 2, 3, ...], [0, 0, 0, 1, 2, ...], ...]
    # This represents the cumulative cost from each state to the leaf node
    shifted_is_solved = jnp.concatenate(
        (is_solved[:, 1:], jnp.zeros((topk_branch_size, 1), dtype=jnp.bool_)), axis=1
    )
    masks = jnp.logical_and(masks, ~shifted_is_solved)
    # masks: [topk_branch_size, max_depth] ,
    # [[False, False, True, True, True, ...], [False, False, False, True, True, ...], ...]

    preprocessed_data = jax.vmap(jax.vmap(heuristic.pre_process, in_axes=(None, 0)))(
        leaf_solve_configs, path_states
    )
    # preprocessed_data: [topk_branch_size, max_depth, ...]
    flattened_preprocessed_data = jnp.reshape(
        preprocessed_data, (topk_branch_size * max_depth, *preprocessed_data.shape[2:])
    )
    flattened_true_costs = jnp.reshape(true_costs, (topk_branch_size * max_depth,))
    flattened_masks = jnp.reshape(masks, (topk_branch_size * max_depth,))
    return flattened_preprocessed_data, flattened_true_costs, flattened_masks


def wbsdai_dataset_builder(
    puzzle: Puzzle,
    heuristic: NeuralHeuristic,
    batch_size: int = 8192,
    max_nodes: int = int(2e7),
    cost_weight: float = 0.8,
    max_depth: int = 100,
    topk_branch_size: int = int(1e3),
    topk_branch_ratio: float = 1.0,  # use all topk_branch_size
    get_dataset_size: int = int(1e6),
) -> Callable:
    """
    wbsdai_builder is a function that returns a partial function of wbsdai.
    """

    astar_fn = astar_builder(
        puzzle, heuristic, batch_size, max_nodes, cost_weight, use_heuristic_params=True
    )

    jitted_get_one_solved_branch_samples = jax.jit(
        partial(
            get_one_solved_branch_samples,
            puzzle,
            heuristic,
            astar_fn,
            max_depth,
            topk_branch_size,
            topk_branch_ratio,
        )
    )

    def get_wbsdai_dataset(
        heuristic_params: jax.tree_util.PyTreeDef,
        key: chex.PRNGKey,
    ):
        dataset_size = 0
        list_preprocessed_data = []
        list_true_costs = []
        while dataset_size < get_dataset_size:
            key, subkey = jax.random.split(key)
            preprocessed_data, true_costs, masks = jitted_get_one_solved_branch_samples(
                heuristic_params, subkey
            )
            list_preprocessed_data.append(preprocessed_data[masks])
            list_true_costs.append(true_costs[masks])
            dataset_size += jnp.sum(masks)
        preprocessed_data = jnp.concatenate(list_preprocessed_data, axis=0)[:get_dataset_size]
        true_costs = jnp.concatenate(list_true_costs, axis=0)[:get_dataset_size]
        return preprocessed_data, true_costs

    return get_wbsdai_dataset
