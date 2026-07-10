import math
from functools import partial

import chex
import jax
import jax.numpy as jnp
import xtructure.numpy as xnp
from puxle import Puzzle

from helpers.rich_progress import trange
from train_util.trajectory_dataset_adapter import (
    trajectory_to_eval_trajectory,
    trajectory_to_transition_dataset,
)


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
    trajectory = puzzle.batched_get_random_trajectory(shuffle_length, shuffle_parallel, key)
    return trajectory_to_transition_dataset(trajectory, dataset_minibatch_size)


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
        flatten_dataset = xnp.concatenate(dataset)
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
    trajectory = puzzle.batched_get_random_trajectory(shuffle_length, 1, key)
    return trajectory_to_eval_trajectory(trajectory)
