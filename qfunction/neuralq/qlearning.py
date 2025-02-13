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
        heuristic_params: jax.tree_util.PyTreeDef,
        states: chex.Array,
        target_qs: chex.Array,
    ):
        q_values, variable_updates = q_fn(
            heuristic_params, states, training=True, mutable=["batch_stats"]
        )
        heuristic_params["batch_stats"] = variable_updates["batch_stats"]
        diff = target_qs - q_values
        loss = jnp.mean(jnp.square(diff))
        return loss, (heuristic_params, diff)

    def qlearning(
        key: chex.PRNGKey,
        dataset: tuple[chex.Array, chex.Array],
        heuristic_params: jax.tree_util.PyTreeDef,
        opt_state: optax.OptState,
    ):
        """
        Q-learning is a heuristic for the sliding puzzle problem.
        """
        states, target_q = dataset
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

        def train_loop(carry, batched_dataset):
            heuristic_params, opt_state = carry
            states, target_q = batched_dataset
            (loss, (heuristic_params, diff)), grads = jax.value_and_grad(
                qlearning_loss, has_aux=True
            )(
                heuristic_params,
                states,
                target_q,
            )
            updates, opt_state = optimizer.update(grads, opt_state, params=heuristic_params)
            heuristic_params = optax.apply_updates(heuristic_params, updates)
            return (heuristic_params, opt_state), (loss, diff)

        (heuristic_params, opt_state), (losses, diffs) = jax.lax.scan(
            train_loop,
            (heuristic_params, opt_state),
            (batched_states, batched_target_q),
        )
        loss = jnp.mean(losses)
        mean_abs_diff = jnp.mean(jnp.abs(diffs))
        return heuristic_params, opt_state, loss, mean_abs_diff, diffs

    return jax.jit(qlearning)


def _get_datasets(
    puzzle: Puzzle,
    preproc_fn: Callable,
    q_fn: Callable,
    create_shuffled_path_fn: Callable,
    minibatch_size: int,
    target_heuristic_params: jax.tree_util.PyTreeDef,
    key: chex.PRNGKey,
):
    solve_configs, shuffled_path, move_costs = create_shuffled_path_fn(key)
    neighbors, _ = jax.vmap(puzzle.get_neighbours)(
        solve_configs, shuffled_path
    )  # [batch_size * shuffle_length, action_shape] [batch_size * shuffle_length, action_shape]
    _, neighbors_neighbor_cost = jax.vmap(
        jax.vmap(puzzle.get_neighbours, in_axes=(None, 0)), in_axes=(0, 0)
    )(
        solve_configs, neighbors
    )  # [batch_size * shuffle_length, action_shape]
    equal = jax.vmap(jax.vmap(puzzle.is_solved, in_axes=(None, 0)), in_axes=(0, 0))(
        solve_configs, neighbors
    )  # if equal, return 0
    preproc_neighbors = jax.vmap(jax.vmap(preproc_fn, in_axes=(None, 0)), in_axes=(0, 0))(
        solve_configs, neighbors
    )
    flatten_neighbors = jnp.reshape(
        preproc_neighbors, (-1, minibatch_size, *preproc_neighbors.shape[2:])
    )
    flatten_neighbors_neighbor_cost = jnp.reshape(
        neighbors_neighbor_cost, (-1, minibatch_size, *neighbors_neighbor_cost.shape[2:])
    )

    def q_scan(_, data):
        neighbors, neighbor_cost = data
        q, _ = q_fn(
            target_heuristic_params, neighbors, training=False, mutable=["batch_stats"]
        )  # [minibatch_size, action_shape]
        min_q = jnp.min(q + neighbor_cost, axis=1)
        return None, min_q

    _, target_q = jax.lax.scan(q_scan, None, (flatten_neighbors, flatten_neighbors_neighbor_cost))
    target_q = jnp.concatenate(target_q, axis=0)
    target_q = jnp.reshape(target_q, (preproc_neighbors.shape[0], -1))
    target_q = jnp.maximum(jnp.where(equal, 0.0, target_q), 0.0)
    states = jax.vmap(preproc_fn)(solve_configs, shuffled_path)
    return states, target_q


def get_dataset_builder(
    puzzle: Puzzle,
    preproc_fn: Callable,
    q_fn: Callable,
    dataset_size: int,
    shuffle_parallel: int,
    shuffle_length: int,
    dataset_minibatch_size: int,
):
    steps = math.ceil(dataset_size / (shuffle_parallel * shuffle_length))
    create_shuffled_path_fn = partial(
        create_shuffled_path, puzzle, shuffle_length, shuffle_parallel, dataset_minibatch_size
    )
    jited_get_datasets = jax.jit(
        partial(
            _get_datasets,
            puzzle,
            preproc_fn,
            q_fn,
            create_shuffled_path_fn,
            dataset_minibatch_size,
        )
    )

    def get_datasets(
        heuristic_params: jax.tree_util.PyTreeDef,
        key: chex.PRNGKey,
    ):
        dataset = []
        for _ in range(steps):
            key, subkey = jax.random.split(key)
            dataset.append(jited_get_datasets(heuristic_params, subkey))
        flatten_dataset = jax.tree_util.tree_map(lambda *xs: jnp.concatenate(xs, axis=0), *dataset)
        assert flatten_dataset[0].shape[0] == dataset_size
        return flatten_dataset

    return get_datasets


def create_shuffled_path(
    puzzle: Puzzle,
    shuffle_length: int,
    shuffle_parallel: int,
    dataset_minibatch_size: int,
    key: chex.PRNGKey,
):
    solve_configs = jax.vmap(puzzle.get_solve_config)(jax.random.split(key, shuffle_parallel))
    targets = solve_configs.TargetState

    def get_trajectory_key(targets: Puzzle.State, key: chex.PRNGKey):
        def _scan(carry, _):
            state, key, move_cost = carry
            neighbor_states, cost = jax.vmap(puzzle.get_neighbours, in_axes=(0, 0))(
                solve_configs, state
            )
            is_target = jax.vmap(jax.vmap(puzzle.is_equal, in_axes=(None, 0)), in_axes=(0, 0))(
                targets, neighbor_states
            )
            filled = jnp.isfinite(cost).astype(jnp.float32)
            filled = jnp.where(is_target, 0.0, filled)
            prob = filled / jnp.sum(filled)
            key, subkey = jax.random.split(key)
            choices = jnp.arange(cost.shape[1])
            idx = jax.vmap(
                lambda key, prob: jax.random.choice(key, choices, p=prob), in_axes=(0, 0)
            )(jax.random.split(subkey, prob.shape[0]), prob)
            next_state = jax.vmap(lambda x, y: x[y], in_axes=(0, 0))(neighbor_states, idx)
            cost = jax.vmap(lambda x, y: x[y], in_axes=(0, 0))(cost, idx)
            move_cost = move_cost + cost
            return (next_state, key, move_cost), (next_state, move_cost)

        _, (moves, move_costs) = jax.lax.scan(
            _scan, (targets, key, jnp.zeros(shuffle_parallel)), None, length=shuffle_length
        )
        return moves, move_costs

    moves, move_costs = get_trajectory_key(targets, key)  # [batch_size, shuffle_length][state...]
    solve_configs = jax.tree_util.tree_map(
        lambda x: jnp.tile(x[:, jnp.newaxis, ...], (1, shuffle_length) + (x.ndim - 1) * (1,)),
        solve_configs,
    )
    solve_configs = jax.tree_util.tree_map(lambda x: x.reshape((-1, *x.shape[2:])), solve_configs)
    moves = jax.tree_util.tree_map(lambda x: x.reshape((-1, *x.shape[2:])), moves)
    move_costs = jnp.reshape(move_costs, (-1))
    solve_configs = solve_configs[:dataset_minibatch_size]
    moves = moves[:dataset_minibatch_size]
    move_costs = move_costs[:dataset_minibatch_size]
    return solve_configs, moves, move_costs
