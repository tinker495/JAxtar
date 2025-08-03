import chex
import jax
import jax.numpy as jnp
import xtructure.numpy as xnp
from puxle import Puzzle


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

    (_, last_state, last_move_cost, _), (
        states,
        move_costs,
        inv_actions,
        action_costs,
    ) = jax.lax.scan(
        _scan,
        (target_states, target_states, jnp.zeros(shuffle_parallel), key),
        None,
        length=shuffle_length,
    )  # [shuffle_length, batch_size, ...]

    states = xnp.concatenate(
        [states, last_state[jnp.newaxis, ...]], axis=0
    )  # [shuffle_length + 1, shuffle_parallel, ...]
    move_costs = jnp.concatenate(
        [move_costs, last_move_cost[jnp.newaxis, ...]], axis=0
    )  # [shuffle_length + 1, shuffle_parallel]

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

    states = xnp.concatenate(
        [states, last_state[jnp.newaxis, ...]], axis=0
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
    include_solved_states: bool,
    key: chex.PRNGKey,
):
    inverse_trajectory = get_random_inverse_trajectory(
        puzzle,
        shuffle_length,
        shuffle_parallel,
        key,
    )

    solve_configs = inverse_trajectory["solve_configs"]
    if include_solved_states:
        states = inverse_trajectory["states"][:-1, ...]  # [shuffle_length, shuffle_parallel, ...]
        move_costs = inverse_trajectory["move_costs"][
            :-1, ...
        ]  # [shuffle_length, shuffle_parallel]
    else:
        states = inverse_trajectory["states"][1:, ...]  # [shuffle_length, shuffle_parallel, ...]
        move_costs = inverse_trajectory["move_costs"][1:, ...]  # [shuffle_length, shuffle_parallel]
    inv_actions = inverse_trajectory["actions"]
    action_costs = inverse_trajectory["action_costs"]

    solve_configs = xnp.tile(solve_configs[jnp.newaxis, ...], (shuffle_length, 1))
    solve_configs = solve_configs.flatten()
    states = states.flatten()
    move_costs = move_costs.flatten()
    inv_actions = inv_actions.flatten()
    action_costs = action_costs.flatten()

    return {
        "solve_configs": solve_configs,
        "states": states,
        "move_costs": move_costs,
        "actions": inv_actions,
        "action_costs": action_costs,
    }


def create_hindsight_target_shuffled_path(
    puzzle: Puzzle,
    shuffle_length: int,
    shuffle_parallel: int,
    include_solved_states: bool,
    key: chex.PRNGKey,
):
    assert not puzzle.fixed_target, "Fixed target is not supported for hindsight target"
    trajectory = get_random_trajectory(puzzle, shuffle_length, shuffle_parallel, key)

    original_solve_configs = trajectory["solve_configs"]  # [shuffle_parallel, ...]
    states = trajectory["states"]  # [shuffle_length + 1, shuffle_parallel, ...]
    move_costs = trajectory["move_costs"]  # [shuffle_length + 1, shuffle_parallel]
    actions = trajectory["actions"]  # [shuffle_length, shuffle_parallel]
    action_costs = trajectory["action_costs"]  # [shuffle_length, shuffle_parallel]

    targets = states[-1, ...]  # [shuffle_parallel, ...]
    if include_solved_states:
        states = states[
            1:, ...
        ]  # [shuffle_length, shuffle_parallel, ...] this is include the last state
    else:
        states = states[
            :-1, ...
        ]  # [shuffle_length, shuffle_parallel, ...] this is exclude the last state

    solve_configs = puzzle.batched_hindsight_transform(
        original_solve_configs, targets
    )  # [shuffle_parallel, ...]
    solve_configs = xnp.tile(solve_configs[jnp.newaxis, ...], (shuffle_length, 1))

    if include_solved_states:
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
    else:
        move_costs = (
            move_costs[-1, ...] - move_costs[:-1, ...]
        )  # [shuffle_length, shuffle_parallel]

    solve_configs = solve_configs.flatten()
    states = states.flatten()
    move_costs = move_costs.flatten()
    actions = actions.flatten()
    action_costs = action_costs.flatten()

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
    include_solved_states: bool,
    key: chex.PRNGKey,
):
    assert not puzzle.fixed_target, "Fixed target is not supported for hindsight target"
    key, subkey = jax.random.split(key)
    trajectory = get_random_trajectory(puzzle, shuffle_length, shuffle_parallel, subkey)

    original_solve_configs = trajectory["solve_configs"]  # [P, ...]
    states = trajectory["states"]  # [L+1, P, ...]
    move_costs = trajectory["move_costs"]  # [L+1, P]
    actions = trajectory["actions"]  # [L, P]
    action_costs = trajectory["action_costs"]  # [L, P]

    # Uniformly sample path length `k` and then a valid starting point `i`.
    # This ensures that the distribution of path lengths `k` is uniform.
    key, key_k, key_i = jax.random.split(key, 3)

    # 1. Sample path lengths `k` uniformly from [0, L] or [1, L]
    minval = 0 if include_solved_states else 1
    k = jax.random.randint(
        key_k,
        shape=(shuffle_length, shuffle_parallel),
        minval=minval,
        maxval=shuffle_length + 1,
    )  # [L, P]

    # 2. For each `k`, sample start_index `i` uniformly from [0, L-k]
    random_floats = jax.random.uniform(key_i, shape=(shuffle_length, shuffle_parallel))  # [L, P]
    max_start_idx = shuffle_length - k
    start_indices = (random_floats * (max_start_idx + 1)).astype(jnp.int32)  # [L, P]

    # 3. The target_index `j` is simply i + k
    target_indices = start_indices + k  # [L, P]

    parallel_indices = jnp.tile(
        jnp.arange(shuffle_parallel)[None, :], (shuffle_length, 1)
    )  # [L, P]

    # Gather data using the sampled indices
    start_states = states[start_indices, parallel_indices]
    target_states = states[target_indices, parallel_indices]

    start_move_costs = move_costs[start_indices, parallel_indices]
    target_move_costs = move_costs[target_indices, parallel_indices]
    final_move_costs = target_move_costs - start_move_costs

    # Handle boundary condition: clamp start_indices to prevent out-of-bounds access
    # actions array has size [L], so valid indices are [0, L-1]
    clamped_start_indices = jnp.clip(start_indices, 0, shuffle_length - 1)
    final_actions = actions[clamped_start_indices, parallel_indices]
    final_action_costs = action_costs[clamped_start_indices, parallel_indices]

    # For cases where start_state == target_state (k=0), set action cost to 0
    # This represents reaching the goal state (no further action needed)
    is_goal_state = (k == 0) & include_solved_states
    final_action_costs = jnp.where(is_goal_state, 0.0, final_action_costs)

    # Apply hindsight transform
    tiled_solve_configs = xnp.tile(original_solve_configs[None, ...], (shuffle_length, 1))

    flat_tiled_sc = tiled_solve_configs.flatten()
    flat_target_states = target_states.flatten()
    final_solve_configs = puzzle.batched_hindsight_transform(flat_tiled_sc, flat_target_states)

    # Flatten the rest of the data
    final_start_states = start_states.flatten()
    final_move_costs = final_move_costs.flatten()
    final_actions = final_actions.flatten()
    final_action_costs = final_action_costs.flatten()

    return {
        "solve_configs": final_solve_configs,
        "states": final_start_states,
        "move_costs": final_move_costs,
        "actions": final_actions,
        "action_costs": final_action_costs,
    }
