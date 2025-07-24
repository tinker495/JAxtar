from functools import partial
from typing import Any, Callable, TypeAlias

import chex
import jax
import jax.numpy as jnp
from puxle import Puzzle

from JAxtar.search_base import Current, HashIdx, Parent, SearchResult

PARAM_DTYPE: TypeAlias = Any


@partial(
    jax.jit,
    static_argnums=(
        1,
        2,
    ),
)
def get_k_promising_branch_paths(
    search_result: SearchResult, optimal_k: int = 1000, max_depth: int = 100
) -> tuple[Current, Parent, chex.Array]:
    """
    Get all branch paths from the solved state.
    All closed states are pseudo-optimal (they are optimal when the heuristic is admissible).
    This allows us to collect ground truth heuristic values from these states.
    If the heuristic is not admissible, the optimality of these paths cannot be guaranteed.
    All closed states are generally close to optimal paths, even if the heuristic is not perfectly admissible.
    """
    # Ensure optimal_k is not larger than the total number of elements
    optimal_k = min(optimal_k, search_result.cost.size)
    closed_masks = jnp.isfinite(search_result.cost)  # [size_table]
    closed_masks = closed_masks.at[-1].set(False)  # mask the last row as a dummy node
    no_parented_masks = (
        jnp.ones_like(closed_masks, dtype=jnp.bool_)
        .at[search_result.parent.hashidx.index]
        .set(False)
    )
    leaf_mask = jnp.logical_and(closed_masks, no_parented_masks)
    masked_cost = jnp.where(leaf_mask, search_result.cost, jnp.inf)  # [size_table]
    flattened_cost = jnp.reshape(masked_cost, (-1,))  # [size_table]
    flattened_idxs = jnp.stack(
        jnp.unravel_index(jnp.arange(search_result.cost.size), search_result.cost.shape), axis=1
    ).astype(jnp.uint32)
    # Get top_k indices using partition, which is more efficient than argsort.
    # We get optimal_k smallest elements, but they are not sorted among themselves.
    top_k_indices = jnp.argpartition(flattened_cost, kth=optimal_k - 1)[:optimal_k]
    # Sort only the top_k elements
    top_k_costs = flattened_cost[top_k_indices]
    sorted_order_in_top_k = jnp.argsort(top_k_costs)
    flattend_sort_indices = top_k_indices[sorted_order_in_top_k]

    sorted_idxs = flattened_idxs[flattend_sort_indices].squeeze(axis=-1)
    sorted_cost = flattened_cost[flattend_sort_indices]
    sorted_mask = leaf_mask[sorted_idxs]
    promising_leaf_nodes = Current(
        hashidx=HashIdx(
            index=sorted_idxs.astype(jnp.uint32),
        ),
        cost=sorted_cost,
    )

    paths, path_masks = jax.vmap(SearchResult._get_path, in_axes=(None, 0, 0, None))(
        search_result, promising_leaf_nodes, sorted_mask, max_depth
    )
    return promising_leaf_nodes, paths, path_masks


def get_one_solved_branch_distance_samples(
    puzzle: Puzzle,
    astar_fn: Callable[[Puzzle.SolveConfig, Puzzle.State, PARAM_DTYPE], SearchResult],
    max_depth: int,
    sample_ratio: float,
    use_promising_branch: bool,
    heuristic_params: PARAM_DTYPE,
    key: chex.PRNGKey,
):
    solve_config, initial_state = puzzle.get_inits(key)

    search_result, leafs, filled = astar_fn(solve_config, initial_state, heuristic_params)
    batch_size = filled.shape[0]

    if use_promising_branch:
        leafs, paths, masks = get_k_promising_branch_paths(
            search_result, optimal_k=batch_size, max_depth=max_depth - 1
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
    leaf_solve_configs = jax.vmap(puzzle.hindsight_transform, in_axes=(None, 0))(
        solve_config, leaf_states
    )  # states -> solve_configs
    # leaf_solve_configs: [topk_branch_size, ...]

    max_leaf_costs = jnp.max(leaf_costs)
    leaf_mask = leaf_costs > max_leaf_costs * (1.0 - sample_ratio)  # batch_size
    masks = jnp.where(leaf_mask[:, jnp.newaxis], masks, False)  # [topk_branch_size, max_depth]

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
    is_solved_shifted = jnp.concatenate(
        (is_solved[:, 1:], jnp.zeros_like(is_solved[:, :1])), axis=1
    )
    masks = jnp.logical_and(masks, ~is_solved_shifted)

    # Flatten solve_configs and states for batch processing
    # First, create a tiled version of leaf_solve_configs.
    # Each leaf in leaf_solve_configs typically has shape (batch_size, *config_dims).
    # We want to tile it to (batch_size, max_depth, *config_dims).
    tiled_leaf_solve_configs = jax.tree_util.tree_map(
        lambda leaf: jnp.tile(
            leaf[:, jnp.newaxis],  # Adds singleton dim: (batch_size, 1, *config_dims)
            (1, max_depth) + (leaf.ndim - 1) * (1,),  # Repeats for the new dim
        ),
        leaf_solve_configs,
    )
    # Now, each leaf in tiled_leaf_solve_configs has shape (batch_size, max_depth, *config_dims)

    flattened_solve_configs = jax.tree_util.tree_map(
        lambda x: jnp.reshape(
            x, (batch_size * max_depth, *x.shape[2:])
        ),  # Reshapes (bs, md, *dims) to (bs*md, *dims)
        tiled_leaf_solve_configs,  # Use the correctly tiled PyTree
    )
    flattened_states = jax.tree_util.tree_map(
        lambda x: jnp.reshape(x, (batch_size * max_depth, *x.shape[2:])), path_states
    )
    flattened_true_costs = jnp.reshape(true_costs, (batch_size * max_depth,)).astype(jnp.bfloat16)
    flattened_masks = jnp.reshape(masks, (batch_size * max_depth,))
    return {
        "solve_configs": flattened_solve_configs,
        "states": flattened_states,
        "true_costs": flattened_true_costs,
        "masks": flattened_masks,
        "solved": search_result.solved,
    }


def get_one_solved_branch_q_samples(
    puzzle: Puzzle,
    qstar_fn: Callable[[Puzzle.SolveConfig, Puzzle.State, PARAM_DTYPE], SearchResult],
    max_depth: int,
    sample_ratio: float,
    use_promising_branch: bool,
    qfunction_params: PARAM_DTYPE,
    key: chex.PRNGKey,
):
    solve_config, initial_state = puzzle.get_inits(key)

    search_result, leafs, filled = qstar_fn(solve_config, initial_state, qfunction_params)
    batch_size = filled.shape[0]

    if use_promising_branch:
        leafs, paths, masks = get_k_promising_branch_paths(
            search_result, optimal_k=batch_size, max_depth=max_depth - 1
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
    leaf_solve_configs = jax.vmap(puzzle.hindsight_transform, in_axes=(None, 0))(
        solve_config, leaf_states
    )  # states -> solve_configs
    # leaf_solve_configs: [topk_branch_size, ...]

    max_leaf_costs = jnp.max(leaf_costs)
    leaf_mask = leaf_costs > max_leaf_costs * (1.0 - sample_ratio)  # batch_size
    masks = jnp.where(leaf_mask[:, jnp.newaxis], masks, False)  # [topk_branch_size, max_depth]

    path_states = search_result.get_state(paths)
    path_actions = paths.action  # [topk_branch_size, max_depth - 1]
    path_states = jax.tree_util.tree_map(
        lambda x, y: jnp.concatenate((x, y), axis=1), leaf_states[:, jnp.newaxis], path_states
    )
    path_actions = jnp.concatenate(
        (jnp.zeros((batch_size, 1), dtype=jnp.uint8), path_actions), axis=1
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

    # The target for Q(s, a) should be V(s'), the value of the state s' reached after action a.
    # In our path structure, from state s_k, we take an action to reach s_{k-1}.
    # So for the sample (s_k, a_k), the target is V(s_{k-1}).
    # `true_costs` stores V(s_k) at index k. We need V(s_{k-1}), which is at index k-1.
    # We can shift the `true_costs` array to align V(s_{k-1}) with index k.
    q_targets = jnp.roll(true_costs, shift=1, axis=1)
    # The value at q_targets[:, 0] is from the end of the array,
    # but the sample for k=0 (the leaf node) is masked out by `~is_solved`.

    masks = jnp.logical_and(masks, ~is_solved)
    # masks: [topk_branch_size, max_depth] ,
    # [[False, False, True, True, True, ...], [False, False, False, True, True, ...], ...]

    # Flatten solve_configs and states for batch processing
    # First, create a tiled version of leaf_solve_configs.
    # Each leaf in leaf_solve_configs typically has shape (batch_size, *config_dims).
    # We want to tile it to (batch_size, max_depth, *config_dims).
    tiled_leaf_solve_configs = jax.tree_util.tree_map(
        lambda leaf: jnp.tile(
            leaf[:, jnp.newaxis],  # Adds singleton dim: (batch_size, 1, *config_dims)
            (1, max_depth) + (leaf.ndim - 1) * (1,),  # Repeats for the new dim
        ),
        leaf_solve_configs,
    )
    # Now, each leaf in tiled_leaf_solve_configs has shape (batch_size, max_depth, *config_dims)

    flattened_solve_configs = jax.tree_util.tree_map(
        lambda x: jnp.reshape(
            x, (batch_size * max_depth, *x.shape[2:])
        ),  # Reshapes (bs, md, *dims) to (bs*md, *dims)
        tiled_leaf_solve_configs,  # Use the correctly tiled PyTree
    )
    flattened_states = jax.tree_util.tree_map(
        lambda x: jnp.reshape(x, (batch_size * max_depth, *x.shape[2:])), path_states
    )
    flattened_actions = jnp.reshape(path_actions, (batch_size * max_depth,))
    flattened_true_costs = jnp.reshape(q_targets, (batch_size * max_depth,)).astype(jnp.bfloat16)
    flattened_masks = jnp.reshape(masks, (batch_size * max_depth,))
    return {
        "solve_configs": flattened_solve_configs,
        "states": flattened_states,
        "actions": flattened_actions,
        "true_costs": flattened_true_costs,
        "masks": flattened_masks,
        "solved": search_result.solved,
    }
