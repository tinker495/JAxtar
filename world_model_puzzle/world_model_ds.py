import math
from functools import partial

import chex
import jax
import jax.numpy as jnp
from puxle import Puzzle

from helpers.rich_progress import trange


def get_world_model_dataset_builder(
    puzzle: Puzzle,
    dataset_size: int,
    shuffle_parallel: int,
    shuffle_length: int,
    dataset_minibatch_size: int,
):
    steps = math.ceil(dataset_size / (shuffle_parallel * shuffle_length))
    create_shuffled_path_fn = partial(
        create_shuffled_path, puzzle, shuffle_length, shuffle_parallel, dataset_minibatch_size
    )

    def get_datasets(
        key: chex.PRNGKey,
    ):
        dataset = []
        for _ in trange(steps):
            key, subkey = jax.random.split(key)
            dataset.append(create_shuffled_path_fn(subkey))
        flatten_dataset = jax.tree_util.tree_map(lambda *xs: jnp.concatenate(xs, axis=0), *dataset)
        assert (
            flatten_dataset[0].shape[0][0] == dataset_size
        ), f"{flatten_dataset[0].shape[0][0]} != {dataset_size}"
        return flatten_dataset

    return get_datasets


def create_shuffled_path(
    puzzle: Puzzle,
    shuffle_length: int,
    shuffle_parallel: int,
    dataset_minibatch_size: int,
    key: chex.PRNGKey,
):
    solve_configs, initial_states = jax.vmap(puzzle.get_inits)(
        jax.random.split(key, shuffle_parallel)
    )

    def get_trajectory_key(initial_states: Puzzle.State, key: chex.PRNGKey):
        def _scan(carry, _):
            state, key = carry
            neighbor_states, cost = jax.vmap(puzzle.get_neighbours, in_axes=(0, 0))(
                solve_configs, state
            )
            key, subkey = jax.random.split(key)
            choices = jnp.arange(cost.shape[1])
            action = jax.vmap(lambda key: jax.random.choice(key, choices), in_axes=(0,))(
                jax.random.split(subkey, cost.shape[0])
            )
            next_state = jax.vmap(lambda x, y: x[y], in_axes=(0, 0))(neighbor_states, action)
            return (next_state, key), (state, action, next_state)

        _, (states, actions, next_states) = jax.lax.scan(
            _scan, (initial_states, key), None, length=shuffle_length
        )
        return states, actions, next_states

    states, actions, next_states = get_trajectory_key(
        initial_states, key
    )  # [batch_size, shuffle_length][state...]
    solve_configs = jax.tree_util.tree_map(
        lambda x: jnp.tile(x[:, jnp.newaxis, ...], (1, shuffle_length) + (x.ndim - 1) * (1,)),
        solve_configs,
    )
    states = jax.tree_util.tree_map(lambda x: x.reshape((-1, *x.shape[2:])), states)
    actions = jax.tree_util.tree_map(lambda x: x.reshape((-1, *x.shape[2:])), actions)
    next_states = jax.tree_util.tree_map(lambda x: x.reshape((-1, *x.shape[2:])), next_states)
    states = states[:dataset_minibatch_size]
    actions = actions[:dataset_minibatch_size]
    next_states = next_states[:dataset_minibatch_size]
    return states, actions, next_states


def get_sample_data_builder(
    puzzle: Puzzle,
    dataset_size: int,
    shuffle_parallel: int,
):
    steps = math.ceil(dataset_size / shuffle_parallel)
    create_sample_data_fn = partial(create_sample_data, puzzle, shuffle_parallel)

    def get_datasets(
        key: chex.PRNGKey,
    ):
        dataset = []
        for _ in trange(steps):
            key, subkey = jax.random.split(key)
            dataset.append(create_sample_data_fn(subkey))
        flatten_dataset = jax.tree_util.tree_map(lambda *xs: jnp.concatenate(xs, axis=0), *dataset)
        assert (
            flatten_dataset[0].shape[0][0] == dataset_size
        ), f"{flatten_dataset[0].shape[0][0]} != {dataset_size}"
        return flatten_dataset

    return get_datasets


def create_sample_data(
    puzzle: Puzzle,
    shuffle_parallel: int,
    key: chex.PRNGKey,
):
    solve_configs, initial_states = jax.vmap(puzzle.get_inits)(
        jax.random.split(key, shuffle_parallel)
    )
    target_states = solve_configs.TargetState
    return target_states, initial_states


def create_eval_trajectory(
    puzzle: Puzzle,
    shuffle_length: int,
    key: chex.PRNGKey,
):
    solve_config, initial_state = puzzle.get_inits(key)

    def _scan(carry, _):
        state, key = carry
        neighbor_states, cost = puzzle.get_neighbours(solve_config, state)
        key, subkey = jax.random.split(key)
        choices = jnp.arange(cost.shape[0])
        action = jax.random.choice(subkey, choices)
        next_state = neighbor_states[action]
        return (next_state, key), (state, action)

    (next_state, _), (states, actions) = jax.lax.scan(
        _scan, (initial_state, key), None, length=shuffle_length
    )
    next_state = next_state[jnp.newaxis, ...]
    states = jax.tree_util.tree_map(
        lambda x, y: jnp.concatenate([x, y], axis=0), states, next_state
    )

    return states, actions
