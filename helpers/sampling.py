import chex
import jax
import jax.numpy as jnp

from helpers.util import flatten_array, flatten_tree
from puzzle.puzzle_base import Puzzle


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
        key, subkey = jax.random.split(key)
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
        key, subkey = jax.random.split(key)
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

    _, (states, move_costs, actions, action_costs) = jax.lax.scan(
        _scan,
        (initial_states, initial_states, jnp.zeros(shuffle_parallel), key),
        None,
        length=shuffle_length + 1,
    )  # [shuffle_length + 1, batch_size, ...]

    return {
        "solve_configs": solve_configs,
        "states": states,
        "move_costs": move_costs,
        "actions": actions,
        "action_costs": action_costs,
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

    solve_configs = trajectory["solve_configs"]
    states = trajectory["states"]
    move_costs = trajectory["move_costs"]
    actions = trajectory["actions"]
    action_costs = trajectory["action_costs"]

    targets = states[-1, ...]  # [batch_size, ...]
    states = states[:-1, ...]  # [shuffle_length, batch_size, ...]
    solve_configs = puzzle.batched_hindsight_transform(solve_configs, targets)  # [batch_size, ...]

    solve_configs = jax.tree_util.tree_map(
        lambda x: jnp.tile(x[jnp.newaxis, ...], (shuffle_length, 1) + (x.ndim - 1) * (1,)),
        solve_configs,
    )  # [shuffle_length, batch_size, ...]

    move_costs = move_costs[-1, ...] - move_costs[:-1, ...]  # [shuffle_length, batch_size]
    actions = actions[:-1, ...]  # [shuffle_length, batch_size]
    action_costs = action_costs[:-1, ...]  # [shuffle_length, batch_size]

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

    solve_configs = trajectory["solve_configs"]
    states = trajectory["states"]
    move_costs = trajectory["move_costs"]
    actions = trajectory["actions"]
    action_costs = trajectory["action_costs"]

    solve_configs = jax.vmap(puzzle.batched_hindsight_transform)(
        solve_configs, states
    )  # [shuffle_length + 1, batch_size, ...]
    move_costs = (
        move_costs[jnp.newaxis, ...]
        - move_costs[  # [1, shuffle_length + 1, batch_size]
            :, jnp.newaxis, ...
        ]  # [shuffle_length + 1, 1, batch_size]
    )  # [shuffle_length + 1, shuffle_length + 1, batch_size]

    solve_configs = jax.tree_util.tree_map(
        lambda x: jnp.tile(x[jnp.newaxis, ...], (shuffle_length + 1, 1) + (x.ndim - 1) * (1,)),
        solve_configs,
    )  # [shuffle_length + 1, shuffle_length + 1, batch_size, ...]
    states = jax.tree_util.tree_map(
        lambda x: jnp.tile(x[:, jnp.newaxis, ...], (1, shuffle_length + 1) + (x.ndim - 1) * (1,)),
        states,
    )  # [shuffle_length + 1, shuffle_length + 1, batch_size, ...]

    # Create an explicit upper triangular mask
    upper_tri_mask = jnp.expand_dims(
        jnp.triu(jnp.ones((shuffle_length + 1, shuffle_length + 1)), k=1), axis=-1
    )  # [shuffle_length + 1, shuffle_length + 1, 1]
    # Combine with positive cost condition

    valid_indices = (move_costs > 0) & (
        upper_tri_mask > 0
    )  # [shuffle_length + 1, shuffle_length + 1, batch_size]
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
