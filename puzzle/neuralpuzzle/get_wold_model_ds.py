import math
from functools import partial

import chex
import jax
import jax.numpy as jnp
import numpy as np

from puzzle.puzzle_base import Puzzle


def get_dataset_builder(
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
    ) -> tuple[list[np.ndarray], jnp.ndarray, list[np.ndarray]]:
        dataset = []
        for _ in range(steps):
            key, subkey = jax.random.split(key)
            dataset.append(create_shuffled_path_fn(subkey))
        flatten_dataset = jax.tree_util.tree_map(lambda *xs: jnp.concatenate(xs, axis=0), *dataset)

        states: Puzzle.State = flatten_dataset[0]
        actions: chex.Array = flatten_dataset[1]
        next_states: Puzzle.State = flatten_dataset[2]
        states_img = states.img()
        next_states_img = next_states.img()
        assert actions.shape[0] == dataset_size
        return states_img, actions, next_states_img

    return get_datasets


def create_shuffled_path(
    puzzle: Puzzle,
    shuffle_length: int,
    shuffle_parallel: int,
    dataset_minibatch_size: int,
    key: chex.PRNGKey,
):
    targets = jax.vmap(puzzle.get_target_state)(jax.random.split(key, shuffle_parallel))

    def get_trajectory_key(target: Puzzle.State, key: chex.PRNGKey):
        def _scan(carry, _):
            state, key = carry
            neighbor_states, cost = puzzle.get_neighbours(state, filled=True)
            is_target = jax.vmap(puzzle.is_equal, in_axes=(None, 0))(target, neighbor_states)
            filled = jnp.isfinite(cost).astype(jnp.float32)
            filled = jnp.where(is_target, 0.0, filled)
            prob = filled / jnp.sum(filled)
            key, subkey = jax.random.split(key)
            action = jax.random.choice(subkey, jnp.arange(cost.shape[0]), p=prob)
            next_state = neighbor_states[action]
            return (next_state, key), (state, action, next_state)

        _, (states, actions, next_states) = jax.lax.scan(
            _scan, (target, key), None, length=shuffle_length
        )
        return states, actions, next_states

    states, actions, next_states = jax.vmap(get_trajectory_key)(
        targets, jax.random.split(key, shuffle_parallel)
    )  # [batch_size, shuffle_length][state...][action...][next_state...]
    tiled_targets = jax.tree_util.tree_map(
        lambda x: jnp.tile(x[:, jnp.newaxis, ...], (1, shuffle_length) + (x.ndim - 1) * (1,)),
        targets,
    )
    tiled_targets = jax.tree_util.tree_map(lambda x: x.reshape((-1, *x.shape[2:])), tiled_targets)
    states = jax.tree_util.tree_map(lambda x: x.reshape((-1, *x.shape[2:])), states)
    actions = jnp.reshape(actions, (-1))
    next_states = jax.tree_util.tree_map(lambda x: x.reshape((-1, *x.shape[2:])), next_states)
    tiled_targets = tiled_targets[:dataset_minibatch_size]
    states = states[:dataset_minibatch_size]
    actions = actions[:dataset_minibatch_size]
    next_states = next_states[:dataset_minibatch_size]
    return tiled_targets, states, actions, next_states
