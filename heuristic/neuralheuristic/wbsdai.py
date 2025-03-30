from functools import partial
from typing import Callable

import chex
import jax
import jax.numpy as jnp

from heuristic.neuralheuristic.neuralheuristic_base import (
    NeuralHeuristicBase as NeuralHeuristic,
)
from JAxtar.astar import astar_builder
from JAxtar.search_base import (
    HASH_POINT_DTYPE,
    HASH_TABLE_IDX_DTYPE,
    Current,
    Parent,
    SearchResult,
)
from puzzle.puzzle_base import Puzzle


@partial(
    jax.jit,
    static_argnums=(
        1,
        2,
    ),
)
def get_top_k_branchs_paths(
    search_result: SearchResult, top_k: int = 1000, max_depth: int = 100
) -> tuple[Current, Parent, chex.Array]:
    """
    Get all branch paths from the solved state.
    All closed states are pseudo-optimal (they are optimal when the heuristic is admissible).
    This allows us to collect ground truth heuristic values from these states.
    If the heuristic is not admissible, the optimality of these paths cannot be guaranteed.
    All closed states are generally close to optimal paths, even if the heuristic is not perfectly admissible.
    """
    closed_masks = jnp.isfinite(search_result.cost)  # [size_table, n_table]
    no_parented_masks = (
        jnp.ones_like(closed_masks, dtype=jnp.bool_)
        .at[search_result.parent.index, search_result.parent.table_index]
        .set(False)
    )
    leaf_mask = jnp.logical_and(closed_masks, no_parented_masks)
    masked_cost = jnp.where(leaf_mask, search_result.cost, 0)  # [size_table, n_table]
    flattened_cost = jnp.reshape(masked_cost, (-1,))  # [size_table * n_table]
    flattened_idxs = jnp.stack(
        jnp.unravel_index(jnp.arange(search_result.cost.size), search_result.cost.shape), axis=1
    ).astype(jnp.uint32)
    flattend_sort_indices = jnp.argsort(flattened_cost, descending=True)
    sorted_idxs = flattened_idxs[flattend_sort_indices]
    sorted_cost = flattened_cost[flattend_sort_indices]
    sorted_mask = leaf_mask[sorted_idxs[:, 0], sorted_idxs[:, 1]]
    sorted_leaf_nodes = Current(
        index=sorted_idxs[:, 0].astype(HASH_POINT_DTYPE),
        table_index=sorted_idxs[:, 1].astype(HASH_TABLE_IDX_DTYPE),
        cost=sorted_cost,
    )

    paths = []
    top_k_leaf_nodes = sorted_leaf_nodes[:top_k]
    top_k_mask = sorted_mask[:top_k]
    paths, path_masks = jax.vmap(SearchResult._get_path, in_axes=(None, 0, 0, None))(
        search_result, top_k_leaf_nodes, top_k_mask, max_depth
    )
    return top_k_leaf_nodes, paths, path_masks


def get_one_solved_branch_samples(
    puzzle: Puzzle,
    heuristic: NeuralHeuristic,
    astar_fn: Callable[[Puzzle.SolveConfig, Puzzle.State, jax.tree_util.PyTreeDef], SearchResult],
    max_depth: int,
    use_topk_branch: bool,
    heuristic_params: jax.tree_util.PyTreeDef,
    key: chex.PRNGKey,
):
    solve_config, initial_state = puzzle.get_inits(key)

    search_result, leafs, filled = astar_fn(solve_config, initial_state, heuristic_params)
    batch_size = filled.shape[0]

    if use_topk_branch:
        leafs, paths, masks = get_top_k_branchs_paths(
            search_result, top_k=batch_size, max_depth=max_depth - 1
        )
    else:
        paths, masks = jax.vmap(SearchResult._get_path, in_axes=(None, 0, 0, None))(
            search_result, leafs, filled, max_depth - 1
        )

    # leafs: [topk_branch_size, ...]
    # paths: [topk_branch_size, max_depth - 1, ...]
    # masks: [topk_branch_size, max_depth - 1]
    masks = jnp.concatenate((jnp.ones(masks.shape[0], dtype=bool)[:, jnp.newaxis], masks), axis=1)

    leaf_states = search_result.get_state(leafs)
    leaf_costs = search_result.get_cost(leafs)
    # leaf_states: [topk_branch_size, ...], leaf_costs: [topk_branch_size]
    leaf_solve_configs = puzzle.batched_hindsight_transform(leaf_states)  # states -> solve_configs
    # leaf_solve_configs: [topk_branch_size, ...]

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
        (is_solved[:, 1:], jnp.zeros((is_solved.shape[0], 1), dtype=jnp.bool_)), axis=1
    )
    masks = jnp.logical_and(masks, ~shifted_is_solved)
    # masks: [topk_branch_size, max_depth] ,
    # [[False, False, True, True, True, ...], [False, False, False, True, True, ...], ...]

    preprocessed_data = jax.vmap(jax.vmap(heuristic.pre_process, in_axes=(None, 0)))(
        leaf_solve_configs, path_states
    )
    # preprocessed_data: [topk_branch_size, max_depth, ...]
    flattened_preprocessed_data = jnp.reshape(
        preprocessed_data, (batch_size * max_depth, *preprocessed_data.shape[2:])
    )
    flattened_true_costs = jnp.reshape(true_costs, (batch_size * max_depth,))
    flattened_masks = jnp.reshape(masks, (batch_size * max_depth,))
    return flattened_preprocessed_data, flattened_true_costs, flattened_masks, search_result.solved


def wbsdai_dataset_builder(
    puzzle: Puzzle,
    heuristic: NeuralHeuristic,
    batch_size: int = 8192,
    max_nodes: int = int(2e6),
    cost_weight: float = 1.0 - 1e-3,
    max_depth: int = 100,
    get_dataset_size: int = int(1e6),
    use_topk_branch: bool = False,
) -> Callable:
    """
    wbsdai_builder is a function that returns a partial function of wbsdai.
    """

    astar_fn = astar_builder(
        puzzle,
        heuristic,
        batch_size,
        max_nodes,
        cost_weight,
        use_heuristic_params=True,
        export_last_pops=True,
    )

    jitted_get_one_solved_branch_samples = jax.jit(
        partial(
            get_one_solved_branch_samples,
            puzzle,
            heuristic,
            astar_fn,
            max_depth,
            use_topk_branch,
        )
    )

    def get_wbsdai_dataset(
        heuristic_params: jax.tree_util.PyTreeDef,
        key: chex.PRNGKey,
    ):
        dataset_size = 0
        list_preprocessed_data = []
        list_true_costs = []
        iter_count = 0
        solved_count = 0
        while dataset_size < get_dataset_size:
            key, subkey = jax.random.split(key)
            preprocessed_data, true_costs, masks, solved = jitted_get_one_solved_branch_samples(
                heuristic_params, subkey
            )
            list_preprocessed_data.append(preprocessed_data[masks])
            list_true_costs.append(true_costs[masks])
            dataset_size += jnp.sum(masks)
            iter_count += 1
            solved_count += jnp.sum(solved)
        preprocessed_data = jnp.concatenate(list_preprocessed_data, axis=0)[:get_dataset_size]
        true_costs = jnp.concatenate(list_true_costs, axis=0)[:get_dataset_size]
        preprocessed_data = jnp.asarray(preprocessed_data).astype(jnp.float32)
        true_costs = jnp.asarray(true_costs).astype(jnp.float32)
        assert (
            preprocessed_data.shape[0] == get_dataset_size
        ), f"{preprocessed_data.shape[0]} != {get_dataset_size}"
        assert (
            true_costs.shape[0] == get_dataset_size
        ), f"{true_costs.shape[0]} != {get_dataset_size}"
        return (preprocessed_data, true_costs), iter_count, solved_count, key

    return get_wbsdai_dataset
