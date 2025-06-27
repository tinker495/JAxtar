from functools import partial
from typing import Any, Callable, TypeAlias

import chex
import jax
import jax.numpy as jnp
import jax.test_util
from puxle import Puzzle

from helpers.util import flatten_array, flatten_tree
from JAxtar.search_base import Current, HashIdx, Parent, SearchResult

PARAM_DTYPE: TypeAlias = Any


def get_random_inverse_trajectory(
    puzzle: Puzzle,
    shuffle_length: int,
    shuffle_parallel: int,
    key: chex.PRNGKey,
):
    solve_configs, _ = jax.vmap(puzzle.get_inits)(jax.random.split(key, shuffle_parallel))
    target_states = jax.vmap(puzzle.solve_config_to_state_transform, in_axes=(0, 0))(
        solve_configs, jax.random.split(key, shuffle_parallel)
    )

    def _scan(carry, _):
        old_state, state, move_cost, key = carry
        key, subkey = jax.random.split(key)
        neighbor_states, cost = puzzle.batched_get_inverse_neighbours(
            solve_configs, state, filleds=jnp.ones_like(move_cost), multi_solve_config=True
        )  # [action, batch, ...]
        is_past = jax.vmap(
            jax.vmap(lambda x, y: x == y, in_axes=(None, 0)), in_axes=(0, 1), out_axes=1
        )(
            old_state, neighbor_states
        )  # [action_size, batch_size]
        is_same = jax.vmap(
            jax.vmap(lambda x, y: x == y, in_axes=(None, 0)), in_axes=(0, 1), out_axes=1
        )(
            state, neighbor_states
        )  # [action_size, batch_size]
        filled = jnp.isfinite(cost).astype(jnp.float32)  # [action, batch]
        filled = jnp.where(is_past, 0.0, filled)  # [action, batch]
        filled = jnp.where(is_same, 0.0, filled)  # [action, batch]
        prob = filled / jnp.sum(filled, axis=0)  # [action, batch]
        choices = jnp.arange(cost.shape[0])  # [action]
        inv_actions = jax.vmap(
            lambda key, prob: jax.random.choice(key, choices, p=prob), in_axes=(0, 1)
        )(
            jax.random.split(subkey, prob.shape[1]), prob
        )  # [batch]
        next_state = jax.vmap(
            lambda ns, i: jax.tree_util.tree_map(lambda x: x[i], ns), in_axes=(1, 0), out_axes=0
        )(
            neighbor_states, inv_actions
        )  # [batch, ...]
        cost = jax.vmap(lambda c, i: c[i], in_axes=(1, 0), out_axes=0)(cost, inv_actions)  # [batch]
        return (
            (state, next_state, move_cost + cost, key),  # carry
            (state, move_cost, inv_actions, cost),  # return
        )

    _, (states, move_costs, inv_actions, action_costs) = jax.lax.scan(
        _scan,
        (target_states, target_states, jnp.zeros(shuffle_parallel), key),
        None,
        length=shuffle_length,
    )  # [shuffle_length, batch_size, ...]
    return {
        "solve_configs": solve_configs,
        "states": states,
        "move_costs": move_costs,
        "actions": inv_actions,
        "action_costs": action_costs,
    }


def get_random_trajectory(
    puzzle: Puzzle,
    shuffle_length: int,
    shuffle_parallel: int,
    key: chex.PRNGKey,
):
    solve_configs, initial_states = jax.vmap(puzzle.get_inits)(
        jax.random.split(key, shuffle_parallel)
    )

    def _scan(carry, _):
        old_state, state, move_cost, key = carry
        key, subkey = jax.random.split(key)
        neighbor_states, cost = puzzle.batched_get_neighbours(
            solve_configs, state, filleds=jnp.ones_like(move_cost), multi_solve_config=True
        )  # [action, batch, ...]
        is_past = jax.vmap(
            jax.vmap(lambda x, y: x == y, in_axes=(None, 0)), in_axes=(0, 1), out_axes=1
        )(
            old_state, neighbor_states
        )  # [action_size, batch_size]
        is_same = jax.vmap(
            jax.vmap(lambda x, y: x == y, in_axes=(None, 0)), in_axes=(0, 1), out_axes=1
        )(
            state, neighbor_states
        )  # [action_size, batch_size]
        filled = jnp.isfinite(cost).astype(jnp.float32)  # [action, batch]
        filled = jnp.where(is_past, 0.0, filled)  # [action, batch]
        filled = jnp.where(is_same, 0.0, filled)  # [action, batch]
        prob = filled / jnp.sum(filled, axis=0)  # [action, batch]
        choices = jnp.arange(cost.shape[0])  # [action]
        actions = jax.vmap(
            lambda key, prob: jax.random.choice(key, choices, p=prob), in_axes=(0, 1)
        )(
            jax.random.split(subkey, prob.shape[1]), prob
        )  # [batch]
        next_state = jax.vmap(
            lambda ns, i: jax.tree_util.tree_map(lambda x: x[i], ns), in_axes=(1, 0), out_axes=0
        )(
            neighbor_states, actions
        )  # [batch, ...]
        cost = jax.vmap(lambda c, i: c[i], in_axes=(1, 0), out_axes=0)(cost, actions)  # [batch]
        return (
            (state, next_state, move_cost + cost, key),  # carry
            (state, move_cost, actions, cost),  # return
        )

    (_, last_state, last_move_cost, _), (states, move_costs, actions, action_costs) = jax.lax.scan(
        _scan,
        (initial_states, initial_states, jnp.zeros(shuffle_parallel), key),
        None,
        length=shuffle_length,
    )  # [shuffle_length, shuffle_parallel, ...]

    states = jax.tree_util.tree_map(
        lambda x, y: jnp.concatenate([x, y[jnp.newaxis, ...]], axis=0), states, last_state
    )  # [shuffle_length + 1, shuffle_parallel, ...]
    move_costs = jnp.concatenate(
        [move_costs, last_move_cost[jnp.newaxis, ...]], axis=0
    )  # [shuffle_length + 1, shuffle_parallel]

    return {
        "solve_configs": solve_configs,  # [shuffle_parallel, ...]
        "states": states,  # [shuffle_length + 1, shuffle_parallel, ...]
        "move_costs": move_costs,  # [shuffle_length + 1, shuffle_parallel]
        "actions": actions,  # [shuffle_length, shuffle_parallel]
        "action_costs": action_costs,  # [shuffle_length, shuffle_parallel]
    }


def create_target_shuffled_path(
    puzzle: Puzzle,
    shuffle_length: int,
    shuffle_parallel: int,
    key: chex.PRNGKey,
):
    inverse_trajectory = get_random_inverse_trajectory(
        puzzle, shuffle_length, shuffle_parallel, key
    )

    solve_configs = inverse_trajectory["solve_configs"]
    states = inverse_trajectory["states"]
    move_costs = inverse_trajectory["move_costs"]
    inv_actions = inverse_trajectory["actions"]
    action_costs = inverse_trajectory["action_costs"]

    solve_configs = jax.tree_util.tree_map(
        lambda x: jnp.tile(x[jnp.newaxis, ...], (shuffle_length, 1) + (x.ndim - 1) * (1,)),
        solve_configs,
    )  # [shuffle_length, batch_size, ...]

    solve_configs = flatten_tree(solve_configs, 2)
    states = flatten_tree(states, 2)
    move_costs = flatten_array(move_costs, 2)
    inv_actions = flatten_array(inv_actions, 2)
    action_costs = flatten_array(action_costs, 2)

    return {
        "solve_configs": solve_configs,
        "states": states,
        "move_costs": move_costs,
        "actions": inv_actions,
        # inverse action is not guaranteed to be valid not fully mapped to original action
        "action_costs": action_costs,
    }


def create_hindsight_target_shuffled_path(
    puzzle: Puzzle,
    shuffle_length: int,
    shuffle_parallel: int,
    key: chex.PRNGKey,
):
    trajectory = get_random_trajectory(puzzle, shuffle_length, shuffle_parallel, key)

    solve_configs = trajectory["solve_configs"]  # [shuffle_parallel, ...]
    states = trajectory["states"]  # [shuffle_length + 1, shuffle_parallel, ...]
    move_costs = trajectory["move_costs"]  # [shuffle_length + 1, shuffle_parallel]
    actions = trajectory["actions"]  # [shuffle_length, shuffle_parallel]
    action_costs = trajectory["action_costs"]  # [shuffle_length, shuffle_parallel]

    targets = states[-1, ...]  # [shuffle_parallel, ...]
    states = states[
        1:, ...
    ]  # [shuffle_length, shuffle_parallel, ...] this is include the last state

    solve_configs = puzzle.batched_hindsight_transform(
        solve_configs, targets
    )  # [shuffle_parallel, ...]
    solve_configs = jax.tree_util.tree_map(
        lambda x: jnp.tile(x[jnp.newaxis, ...], (shuffle_length, 1) + (x.ndim - 1) * (1,)),
        solve_configs,
    )  # [shuffle_length, shuffle_parallel, ...]

    move_costs = move_costs[-1, ...] - move_costs[1:, ...]  # [shuffle_length, shuffle_parallel]
    actions = jnp.concatenate(
        [
            actions[1:],
            jax.random.randint(key, (1, shuffle_parallel), minval=0, maxval=puzzle.action_size),
        ]
    )  # [shuffle_length, shuffle_parallel]
    action_costs = jnp.concatenate(
        [action_costs[1:], jnp.zeros((1, shuffle_parallel))]
    )  # [shuffle_length, shuffle_parallel]

    solve_configs = flatten_tree(solve_configs, 2)
    states = flatten_tree(states, 2)
    move_costs = flatten_array(move_costs, 2)
    actions = flatten_array(actions, 2)
    action_costs = flatten_array(action_costs, 2)

    return {
        "solve_configs": solve_configs,
        "states": states,
        "move_costs": move_costs,
        "actions": actions,
        "action_costs": action_costs,
    }


def create_hindsight_target_triangular_shuffled_path(
    puzzle: Puzzle,
    shuffle_length: int,
    shuffle_parallel: int,
    key: chex.PRNGKey,
):
    trajectory = get_random_trajectory(puzzle, shuffle_length, shuffle_parallel, key)

    solve_configs = trajectory["solve_configs"]  # [shuffle_parallel, ...]
    states = trajectory["states"]  # [shuffle_length + 1, shuffle_parallel, ...]
    move_costs = trajectory["move_costs"]  # [shuffle_length + 1, shuffle_parallel]
    actions = trajectory["actions"]  # [shuffle_length, shuffle_parallel]
    action_costs = trajectory["action_costs"]  # [shuffle_length, shuffle_parallel]

    solve_configs = jax.vmap(puzzle.batched_hindsight_transform)(
        solve_configs, states
    )  # [shuffle_length + 1, shuffle_parallel, ...]
    move_costs = (
        move_costs[jnp.newaxis, ...]
        - move_costs[  # [1, shuffle_length + 1, shuffle_parallel]
            :, jnp.newaxis, ...
        ]  # [shuffle_length + 1, 1, shuffle_parallel]
    )  # [shuffle_length + 1, shuffle_length + 1, shuffle_parallel]

    solve_configs = jax.tree_util.tree_map(
        lambda x: jnp.tile(x[jnp.newaxis, ...], (shuffle_length + 1, 1) + (x.ndim - 1) * (1,)),
        solve_configs,
    )  # [shuffle_length + 1, shuffle_length + 1, shuffle_parallel, ...]
    states = jax.tree_util.tree_map(
        lambda x: jnp.tile(x[:, jnp.newaxis, ...], (1, shuffle_length + 1) + (x.ndim - 1) * (1,)),
        states,
    )  # [shuffle_length + 1, shuffle_length + 1, shuffle_parallel, ...]

    # Create an explicit upper triangular mask
    upper_tri_mask = jnp.expand_dims(
        jnp.tril(jnp.ones((shuffle_length + 1, shuffle_length + 1)), k=1), axis=-1
    )  # [shuffle_length + 1, shuffle_length + 1, 1]
    # Combine with positive cost condition

    valid_indices = (move_costs > 0) & (
        upper_tri_mask > 0
    )  # [shuffle_length + 1, shuffle_length + 1, shuffle_parallel]
    idxs = jnp.where(
        valid_indices, size=(shuffle_length * (shuffle_length + 1) // 2 * shuffle_parallel)
    )  # [shuffle_length * (shuffle_length + 1) // 2 * shuffle_parallel]

    solve_configs = solve_configs[
        idxs[0], idxs[1], idxs[2], ...
    ]  # [shuffle_length * (shuffle_length + 1) // 2 * shuffle_parallel, ...]
    states = states[
        idxs[0], idxs[1], idxs[2], ...
    ]  # [shuffle_length * (shuffle_length + 1) // 2 * shuffle_parallel, ...]
    move_costs = move_costs[
        idxs[0], idxs[1], idxs[2]
    ]  # [shuffle_length * (shuffle_length + 1) // 2 * shuffle_parallel]
    actions = actions[
        idxs[0], idxs[1], idxs[2]
    ]  # [shuffle_length * (shuffle_length + 1) // 2 * shuffle_parallel]
    action_costs = action_costs[
        idxs[0], idxs[1], idxs[2]
    ]  # [shuffle_length * (shuffle_length + 1) // 2 * shuffle_parallel]

    return {
        "solve_configs": solve_configs,
        "states": states,
        "move_costs": move_costs,
        "actions": actions,
        "action_costs": action_costs,
    }


@partial(
    jax.jit,
    static_argnums=(
        1,
        2,
    ),
)
def get_k_optimal_branchs_paths(
    search_result: SearchResult, optimal_k: int = 1000, max_depth: int = 100
) -> tuple[Current, Parent, chex.Array]:
    """
    Get all branch paths from the solved state.
    All closed states are pseudo-optimal (they are optimal when the heuristic is admissible).
    This allows us to collect ground truth heuristic values from these states.
    If the heuristic is not admissible, the optimality of these paths cannot be guaranteed.
    All closed states are generally close to optimal paths, even if the heuristic is not perfectly admissible.
    """
    closed_masks = jnp.isfinite(search_result.cost)  # [size_table, n_table]
    closed_masks = closed_masks.at[-1, :].set(False)  # mask the last row as a dummy node
    no_parented_masks = (
        jnp.ones_like(closed_masks, dtype=jnp.bool_)
        .at[search_result.parent.index, search_result.parent.table_index]
        .set(False)
    )
    leaf_mask = jnp.logical_and(closed_masks, no_parented_masks)
    masked_cost = jnp.where(leaf_mask, search_result.cost, jnp.inf)  # [size_table, n_table]
    masked_dist = jnp.where(leaf_mask, search_result.dist, jnp.inf)  # [size_table, n_table]
    masked_sum = masked_cost + masked_dist  # [size_table, n_table]
    flattened_cost = jnp.reshape(masked_sum, (-1,))  # [size_table * n_table]
    flattened_idxs = jnp.stack(
        jnp.unravel_index(jnp.arange(search_result.cost.size), search_result.cost.shape), axis=1
    ).astype(jnp.uint32)
    flattend_sort_indices = jnp.argsort(flattened_cost, descending=False)
    sorted_idxs = flattened_idxs[flattend_sort_indices]
    sorted_cost = flattened_cost[flattend_sort_indices]
    sorted_mask = leaf_mask[sorted_idxs[:, 0], sorted_idxs[:, 1]]
    sorted_leaf_nodes = Current(
        hashidx=HashIdx(
            index=sorted_idxs[:, 0].astype(jnp.uint32),
            table_index=sorted_idxs[:, 1].astype(jnp.uint8),
        ),
        cost=sorted_cost,
    )

    optimal_k_leaf_nodes = sorted_leaf_nodes[:optimal_k]
    optimal_k_mask = sorted_mask[:optimal_k]
    paths, path_masks = jax.vmap(SearchResult._get_path, in_axes=(None, 0, 0, None))(
        search_result, optimal_k_leaf_nodes, optimal_k_mask, max_depth
    )
    return optimal_k_leaf_nodes, paths, path_masks


def get_one_solved_branch_distance_samples(
    puzzle: Puzzle,
    astar_fn: Callable[[Puzzle.SolveConfig, Puzzle.State, PARAM_DTYPE], SearchResult],
    max_depth: int,
    sample_ratio: float,
    use_optimal_branch: bool,
    heuristic_params: PARAM_DTYPE,
    key: chex.PRNGKey,
):
    solve_config, initial_state = puzzle.get_inits(key)

    search_result, leafs, filled = astar_fn(solve_config, initial_state, heuristic_params)
    batch_size = filled.shape[0]

    if use_optimal_branch:
        leafs, paths, masks = get_k_optimal_branchs_paths(
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
    leaf_solve_configs = puzzle.batched_hindsight_transform(
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
    shifted_is_solved = jnp.concatenate(
        (is_solved[:, 1:], jnp.zeros((is_solved.shape[0], 1), dtype=jnp.bool_)), axis=1
    )
    masks = jnp.logical_and(masks, ~shifted_is_solved)
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
    flattened_true_costs = jnp.reshape(true_costs, (batch_size * max_depth,)).astype(jnp.bfloat16)
    flattened_masks = jnp.reshape(masks, (batch_size * max_depth,))
    return (
        flattened_solve_configs,
        flattened_states,
        flattened_true_costs,
        flattened_masks,
        search_result.solved,
    )


def get_one_solved_branch_q_samples(
    puzzle: Puzzle,
    qstar_fn: Callable[[Puzzle.SolveConfig, Puzzle.State, PARAM_DTYPE], SearchResult],
    max_depth: int,
    sample_ratio: float,
    use_optimal_branch: bool,
    qfunction_params: PARAM_DTYPE,
    key: chex.PRNGKey,
):
    solve_config, initial_state = puzzle.get_inits(key)

    search_result, leafs, filled = qstar_fn(solve_config, initial_state, qfunction_params)
    batch_size = filled.shape[0]

    if use_optimal_branch:
        leafs, paths, masks = get_k_optimal_branchs_paths(
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
    leaf_solve_configs = puzzle.batched_hindsight_transform(
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
    true_costs = jnp.roll(true_costs, 1, axis=1)
    true_costs = true_costs.at[:, 0].set(0)
    # true_costs: [topk_branch_size, max_depth] - Array of cumulative costs from each state to leaf node
    # Example after roll: [[0, 0, 0, 1, 2, ...], [0, 0, 0, 0, 1, ...], ...] where each row represents a branch
    # This represents the cumulative cost from each state to the leaf node
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
    flattened_true_costs = jnp.reshape(true_costs, (batch_size * max_depth,)).astype(jnp.bfloat16)
    flattened_masks = jnp.reshape(masks, (batch_size * max_depth,))
    return (
        flattened_solve_configs,
        flattened_states,
        flattened_actions,
        flattened_true_costs,
        flattened_masks,
        search_result.solved,
    )
