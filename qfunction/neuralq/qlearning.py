import math
from functools import partial
from typing import Callable

import chex
import jax
import jax.numpy as jnp
import optax

from puzzle.puzzle_base import Puzzle


def qlearning_builder(
    minibatch_size: int,
    q_fn: Callable,
    optimizer: optax.GradientTransformation,
):
    def qlearning_loss(
        q_params: jax.tree_util.PyTreeDef,
        states: chex.Array,
        actions: chex.Array,
        target_qs: chex.Array,
    ):
        q_values, variable_updates = q_fn(q_params, states, training=True, mutable=["batch_stats"])
        new_params = {"params": q_params["params"], "batch_stats": variable_updates["batch_stats"]}
        q_values_at_actions = jnp.take_along_axis(q_values, actions[:, jnp.newaxis], axis=1)
        diff = target_qs.squeeze() - q_values_at_actions.squeeze()
        loss = jnp.mean(jnp.square(diff))
        return loss, (new_params, diff)

    def qlearning(
        key: chex.PRNGKey,
        dataset: tuple[chex.Array, chex.Array, chex.Array],
        q_params: jax.tree_util.PyTreeDef,
        opt_state: optax.OptState,
    ):
        """
        Q-learning is a heuristic for the sliding puzzle problem.
        """
        states, target_q, actions = dataset
        data_size = target_q.shape[0]
        batch_size = math.ceil(data_size / minibatch_size)

        batch_indexs = jnp.concatenate(
            [
                jax.random.permutation(key, jnp.arange(data_size)),
                jax.random.randint(key, (batch_size * minibatch_size - data_size,), 0, data_size),
            ],
            axis=0,
        )  # [batch_size * minibatch_size]
        batch_indexs = jnp.reshape(batch_indexs, (batch_size, minibatch_size))

        batched_states = jnp.take(states, batch_indexs, axis=0)
        batched_target_q = jnp.take(target_q, batch_indexs, axis=0)
        batched_actions = jnp.take(actions, batch_indexs, axis=0)

        def train_loop(carry, batched_dataset):
            q_params, opt_state = carry
            states, target_q, actions = batched_dataset
            (loss, (q_params, diff)), grads = jax.value_and_grad(qlearning_loss, has_aux=True)(
                q_params,
                states,
                actions,
                target_q,
            )
            updates, opt_state = optimizer.update(grads, opt_state, params=q_params)
            q_params = optax.apply_updates(q_params, updates)
            return (q_params, opt_state), (loss, diff)

        (q_params, opt_state), (losses, diffs) = jax.lax.scan(
            train_loop,
            (q_params, opt_state),
            (batched_states, batched_target_q, batched_actions),
        )
        loss = jnp.mean(losses)
        mean_abs_diff = jnp.mean(jnp.abs(diffs))
        return q_params, opt_state, loss, mean_abs_diff, diffs

    return jax.jit(qlearning)


def boltzmann_action_selection(q_values: chex.Array, temperature: float = 1.0 / 3.0) -> chex.Array:
    q_values = -q_values / temperature
    q_values = jnp.exp(q_values)
    probs = q_values / jnp.sum(q_values, axis=1, keepdims=True)
    return probs


def _get_datasets(
    puzzle: Puzzle,
    preproc_fn: Callable,
    q_fn: Callable,
    minibatch_size: int,
    target_q_params: jax.tree_util.PyTreeDef,
    q_params: jax.tree_util.PyTreeDef,
    shuffled_path: tuple[Puzzle.SolveConfig, Puzzle.State, chex.Array],
    key: chex.PRNGKey,
):
    solve_configs, shuffled_path, move_costs = shuffled_path

    minibatched_solve_configs = jax.tree_util.tree_map(
        lambda x: x.reshape((-1, minibatch_size, *x.shape[1:])), solve_configs
    )
    minibatched_shuffled_path = jax.tree_util.tree_map(
        lambda x: x.reshape((-1, minibatch_size, *x.shape[1:])), shuffled_path
    )
    minibatched_move_costs = jnp.reshape(move_costs, (-1, minibatch_size))

    def get_minibatched_datasets(_, vals):
        solve_configs, shuffled_path, move_costs = vals
        solved = puzzle.batched_is_solved(
            solve_configs, shuffled_path, multi_solve_config=True
        )  # [batch_size]

        path_preproc = jax.vmap(preproc_fn, in_axes=(0, 0))(solve_configs, shuffled_path)
        q_values, _ = q_fn(q_params, path_preproc, training=False, mutable=["batch_stats"])
        probs = boltzmann_action_selection(q_values)
        idxs = jnp.arange(q_values.shape[1])  # action_size
        actions = jax.vmap(lambda key, p: jax.random.choice(key, idxs, p=p), in_axes=(0, 0))(
            jax.random.split(key, q_values.shape[0]), probs
        )

        neighbors, cost = puzzle.batched_get_neighbours(
            solve_configs, shuffled_path, filleds=jnp.ones_like(move_costs), multi_solve_config=True
        )  # [action_size, batch_size] [action_size, batch_size]
        batch_size = actions.shape[0]
        selected_neighbors = jax.tree_util.tree_map(
            lambda x: x[actions, jnp.arange(batch_size), :],
            neighbors,
        )
        selected_costs = jnp.take_along_axis(cost, actions[jnp.newaxis, :], axis=0).squeeze(0)
        selected_neighbors_solved = puzzle.batched_is_solved(
            solve_configs, selected_neighbors, multi_solve_config=True
        )

        preproc_neighbors = jax.vmap(preproc_fn, in_axes=(0, 0))(solve_configs, selected_neighbors)

        q, _ = q_fn(
            target_q_params, preproc_neighbors, training=False, mutable=["batch_stats"]
        )  # [minibatch_size, action_shape]
        target_q = jnp.min(q, axis=1) + selected_costs
        target_q = jnp.maximum(jnp.where(selected_neighbors_solved, 0.0, target_q), 0.0)
        target_q = jnp.round(target_q)
        target_q = jnp.where(
            solved, 0.0, target_q
        )  # if the puzzle is already solved, the all q is 0

        # target_heuristic must be less than the number of moves
        # it just doesn't make sense to have a heuristic greater than the number of moves
        # heuristic's definition is the optimal cost to reach the target state
        # so it doesn't make sense to have a heuristic greater than the number of moves
        # Q-learning needs to predict values larger than move_costs based on actions
        # because it needs to account for future costs beyond just the immediate move
        # target_q = jnp.minimum(target_q, move_costs + selected_costs)
        states = jax.vmap(preproc_fn)(solve_configs, shuffled_path)
        return None, (states, target_q, actions)

    _, (states, target_q, actions) = jax.lax.scan(
        get_minibatched_datasets,
        None,
        (minibatched_solve_configs, minibatched_shuffled_path, minibatched_move_costs),
    )

    states = states.reshape((-1, *states.shape[2:]))
    target_q = target_q.reshape((-1, *target_q.shape[2:]))
    actions = actions.reshape((-1, *actions.shape[2:]))

    return states, target_q, actions


def get_qlearning_dataset_builder(
    puzzle: Puzzle,
    preproc_fn: Callable,
    q_fn: Callable,
    dataset_size: int,
    shuffle_length: int,
    dataset_minibatch_size: int,
    using_hindsight_target: bool = True,
):
    shuffle_parallel = int(min(math.ceil(dataset_size / shuffle_length), dataset_minibatch_size))
    steps = math.ceil(dataset_size / (shuffle_parallel * shuffle_length))
    shuffle_size = dataset_size // steps

    if using_hindsight_target:
        create_shuffled_path_fn = partial(
            create_hindsight_target_shuffled_path,
            puzzle,
            shuffle_length,
            shuffle_parallel,
            shuffle_size,
        )
    else:
        create_shuffled_path_fn = partial(
            create_target_shuffled_path,
            puzzle,
            shuffle_length,
            shuffle_parallel,
            shuffle_size,
        )

    jited_create_shuffled_path = jax.jit(create_shuffled_path_fn)

    jited_get_datasets = jax.jit(
        partial(
            _get_datasets,
            puzzle,
            preproc_fn,
            q_fn,
            dataset_minibatch_size,
        )
    )

    def get_datasets(
        target_q_params: jax.tree_util.PyTreeDef,
        q_params: jax.tree_util.PyTreeDef,
        key: chex.PRNGKey,
    ):
        paths = []
        for _ in range(steps):
            key, subkey = jax.random.split(key)
            paths.append(jited_create_shuffled_path(subkey))
        paths = jax.tree_util.tree_map(lambda *xs: jnp.concatenate(xs, axis=0), *paths)

        flatten_dataset = jited_get_datasets(target_q_params, q_params, paths, key)
        assert (
            flatten_dataset[0].shape[0] == dataset_size
        ), f"{flatten_dataset[0].shape[0]} != {dataset_size}"
        return flatten_dataset

    return get_datasets


def create_target_shuffled_path(
    puzzle: Puzzle,
    shuffle_length: int,
    shuffle_parallel: int,
    shuffle_size: int,
    key: chex.PRNGKey,
):
    solve_configs, _ = jax.vmap(puzzle.get_inits)(jax.random.split(key, shuffle_parallel))
    targets = jax.vmap(puzzle.solve_config_to_state_transform, in_axes=(0, 0))(
        solve_configs, jax.random.split(key, shuffle_parallel)
    )

    def _scan(carry, _):
        old_state, state, key, move_cost = carry
        neighbor_states, cost = puzzle.batched_get_inverse_neighbours(
            solve_configs, state, filleds=jnp.ones_like(move_cost), multi_solve_config=True
        )  # [action, batch, ...]
        is_past = jax.vmap(
            jax.vmap(puzzle.is_equal, in_axes=(None, 0)), in_axes=(0, 1), out_axes=1
        )(
            old_state, neighbor_states
        )  # [action_size, batch_size]
        is_same = jax.vmap(
            jax.vmap(puzzle.is_equal, in_axes=(None, 0)), in_axes=(0, 1), out_axes=1
        )(
            state, neighbor_states
        )  # [action_size, batch_size]
        filled = jnp.isfinite(cost).astype(jnp.float32)  # [action, batch]
        filled = jnp.where(is_past, 0.0, filled)  # [action, batch]
        filled = jnp.where(is_same, 0.0, filled)  # [action, batch]
        prob = filled / jnp.sum(filled, axis=0)  # [action, batch]
        key, subkey = jax.random.split(key)
        choices = jnp.arange(cost.shape[0])  # [action]
        idx = jax.vmap(lambda key, prob: jax.random.choice(key, choices, p=prob), in_axes=(0, 1))(
            jax.random.split(subkey, prob.shape[1]), prob
        )  # [batch]
        next_state = jax.vmap(
            lambda ns, i: jax.tree_util.tree_map(lambda x: x[i], ns), in_axes=(1, 0), out_axes=0
        )(
            neighbor_states, idx
        )  # [batch, ...]
        cost = jax.vmap(lambda c, i: c[i], in_axes=(1, 0), out_axes=0)(cost, idx)  # [batch]
        move_cost = move_cost + cost
        return (state, next_state, key, move_cost), (next_state, move_cost)

    _, (moves, move_costs) = jax.lax.scan(
        _scan, (targets, targets, key, jnp.zeros(shuffle_parallel)), None, length=shuffle_length
    )  # [batch_size, shuffle_length, ...]
    moves = jax.tree_util.tree_map(
        lambda x: jnp.swapaxes(x, 0, 1), moves
    )  # [shuffle_length, batch_size, ...]
    move_costs = jnp.swapaxes(move_costs, 0, 1)  # [shuffle_length, batch_size]

    solve_configs = jax.tree_util.tree_map(
        lambda x: jnp.tile(x[:, jnp.newaxis, ...], (1, shuffle_length) + (x.ndim - 1) * (1,)),
        solve_configs,
    )  # [batch_size, shuffle_length, ...]
    solve_configs = jax.tree_util.tree_map(
        lambda x: x.reshape((-1, *x.shape[2:])), solve_configs
    )  # [batch_size * shuffle_length, ...]
    moves = jax.tree_util.tree_map(
        lambda x: x.reshape((-1, *x.shape[2:])), moves
    )  # [batch_size * shuffle_length, ...]
    move_costs = jnp.reshape(move_costs, (-1))  # [batch_size * shuffle_length]
    solve_configs = solve_configs[:shuffle_size]
    moves = moves[:shuffle_size]
    move_costs = move_costs[:shuffle_size]
    return solve_configs, moves, move_costs


def create_hindsight_target_shuffled_path(
    puzzle: Puzzle,
    shuffle_length: int,
    shuffle_parallel: int,
    shuffle_size: int,
    key: chex.PRNGKey,
):
    solve_configs, initial_states = jax.vmap(puzzle.get_inits)(
        jax.random.split(key, shuffle_parallel)
    )

    def _scan(carry, _):
        old_state, state, key, move_cost = carry
        neighbor_states, cost = puzzle.batched_get_neighbours(
            solve_configs, state, filleds=jnp.ones_like(move_cost), multi_solve_config=True
        )  # [action, batch, ...]
        is_past = jax.vmap(
            jax.vmap(puzzle.is_equal, in_axes=(None, 0)), in_axes=(0, 1), out_axes=1
        )(
            old_state, neighbor_states
        )  # [action_size, batch_size]
        is_same = jax.vmap(
            jax.vmap(puzzle.is_equal, in_axes=(None, 0)), in_axes=(0, 1), out_axes=1
        )(
            state, neighbor_states
        )  # [action_size, batch_size]
        filled = jnp.isfinite(cost).astype(jnp.float32)  # [action, batch]
        filled = jnp.where(is_past, 0.0, filled)  # [action, batch]
        filled = jnp.where(is_same, 0.0, filled)  # [action, batch]
        prob = filled / jnp.sum(filled, axis=0)  # [action, batch]
        key, subkey = jax.random.split(key)
        choices = jnp.arange(cost.shape[0])  # [action]
        idx = jax.vmap(lambda key, prob: jax.random.choice(key, choices, p=prob), in_axes=(0, 1))(
            jax.random.split(subkey, prob.shape[1]), prob
        )  # [batch]
        next_state = jax.vmap(
            lambda ns, i: jax.tree_util.tree_map(lambda x: x[i], ns), in_axes=(1, 0), out_axes=0
        )(
            neighbor_states, idx
        )  # [batch, ...]
        cost = jax.vmap(lambda c, i: c[i], in_axes=(1, 0), out_axes=0)(cost, idx)  # [batch]
        move_cost = move_cost + cost
        return (state, next_state, key, move_cost), (next_state, move_cost)

    _, (moves, move_costs) = jax.lax.scan(
        _scan,
        (initial_states, initial_states, key, jnp.zeros(shuffle_parallel)),
        None,
        length=shuffle_length + 1,
    )  # [shuffle_length, batch_size, ...]
    solve_configs = puzzle.batched_hindsight_transform(moves[-1, ...])

    move_costs = move_costs[-1] - move_costs

    moves = moves[:-1, ...]
    move_costs = move_costs[:-1, ...]

    moves = jax.tree_util.tree_map(
        lambda x: jnp.swapaxes(x, 0, 1), moves
    )  # [batch_size, shuffle_length, ...]
    move_costs = jnp.swapaxes(move_costs, 0, 1)  # [shuffle_length, batch_size]

    solve_configs = jax.tree_util.tree_map(
        lambda x: jnp.tile(x[:, jnp.newaxis, ...], (1, shuffle_length) + (x.ndim - 1) * (1,)),
        solve_configs,
    )
    solve_configs = jax.tree_util.tree_map(lambda x: x.reshape((-1, *x.shape[2:])), solve_configs)
    moves = jax.tree_util.tree_map(lambda x: x.reshape((-1, *x.shape[2:])), moves)
    move_costs = jnp.reshape(move_costs, (-1))
    solve_configs = solve_configs[:shuffle_size]
    moves = moves[:shuffle_size]
    move_costs = move_costs[:shuffle_size]
    return solve_configs, moves, move_costs
