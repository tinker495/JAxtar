from functools import partial
from typing import Callable

import chex
import jax
import jax.numpy as jnp
import optax
import math
from puzzle.puzzle_base import Puzzle

from tqdm import trange

def davi_builder(
    minibatch_size: int,
    heuristic_fn: Callable,
    heuristic_params: jax.tree_util.PyTreeDef,
):

    def davi_loss(
        heuristic_params: jax.tree_util.PyTreeDef,
        states: chex.Array,
        target_heuristic: chex.Array,
    ):
        current_heuristic = heuristic_fn(heuristic_params, states).squeeze()
        diff = target_heuristic - current_heuristic
        loss = jnp.mean(jnp.square(diff))
        return loss

    optimizer = optax.adabelief(1e-4)
    opt_state = optimizer.init(heuristic_params)

    def davi(
        key: chex.PRNGKey,
        dataset: tuple[chex.Array, chex.Array],
        heuristic_params: jax.tree_util.PyTreeDef,
        opt_state: optax.OptState,
    ):
        """
        DAVI is a heuristic for the sliding puzzle problem.
        """
        states, target_heuristic = dataset
        data_size = target_heuristic.shape[0]
        batch_size = math.ceil(data_size / minibatch_size)

        batch_indexs = jnp.concatenate(
            [
                jax.random.permutation(key, jnp.arange(data_size)),
                jax.random.randint(key, (batch_size * minibatch_size - data_size,), 0, data_size)
            ],
            axis=0
        ) # [batch_size * minibatch_size]
        batch_indexs = jnp.reshape(batch_indexs, (batch_size, minibatch_size))

        batched_states = jnp.take(states, batch_indexs, axis=0)
        batched_target_heuristic = jnp.take(target_heuristic, batch_indexs, axis=0)

        def train_loop(carry, batched_dataset):
            heuristic_params, opt_state = carry
            states, target_heuristic = batched_dataset
            loss, grads = jax.value_and_grad(davi_loss)(
                heuristic_params,
                states,
                target_heuristic,
            )
            updates, opt_state = optimizer.update(grads, opt_state, params=heuristic_params)
            heuristic_params = optax.apply_updates(heuristic_params, updates)
            return (heuristic_params, opt_state), loss

        (heuristic_params, opt_state), losses = jax.lax.scan(
            train_loop, (heuristic_params, opt_state), (batched_states, batched_target_heuristic)
        )
        loss = jnp.mean(losses)
        return heuristic_params, opt_state, loss

    return jax.jit(davi), opt_state

def _get_datasets(
    puzzle: Puzzle,
    preproc_fn: Callable,
    heuristic_fn: Callable,
    create_shuffled_path_fn: Callable,
    key: chex.PRNGKey,
):
    tiled_targets, shuffled_path = create_shuffled_path_fn(
        key
    )  # [batch_size, shuffle_length] [batch_size, shuffle_length]
    neighbors, cost = jax.vmap(puzzle.get_neighbours)(
        shuffled_path
    )  # [batch_size, shuffle_length, 4] [batch_size, shuffle_length, 4]
    equal = jax.vmap(jax.vmap(puzzle.is_solved, in_axes=(0, None)), in_axes=(0, 0))(neighbors, tiled_targets)  # if equal, return 0
    preproc_neighbors = jax.vmap(jax.vmap(preproc_fn, in_axes=(0, None)), in_axes=(0, 0))(neighbors, tiled_targets).squeeze()
    heur = jax.vmap(heuristic_fn)(preproc_neighbors).squeeze()
    heur = jnp.maximum(jnp.where(equal, 0.0, heur), 0.0)
    target_heuristic = jnp.min(heur + cost, axis=1)
    states = jax.vmap(preproc_fn)(shuffled_path, tiled_targets).squeeze()
    return states, target_heuristic

def get_datasets(puzzle: Puzzle, preproc_fn: Callable, heuristic_fn: Callable, target_heuristic_params: jax.tree_util.PyTreeDef, dataset_size: int, batch_size: int, shuffle_length: int, key: chex.PRNGKey):
    steps = math.ceil(dataset_size / (batch_size * shuffle_length))
    create_shuffled_path_fn = partial(
        create_shuffled_path, puzzle, shuffle_length, batch_size
    )
    heuristic_fn = partial(heuristic_fn, target_heuristic_params)
    jited_get_datasets = jax.jit(partial(_get_datasets, puzzle, preproc_fn, heuristic_fn, create_shuffled_path_fn))
    dataset = []
    for _ in trange(steps):
        dataset.append(jited_get_datasets(key))
    flatten_dataset = jax.tree_util.tree_map(
        lambda *xs: jnp.concatenate(xs, axis=0), *dataset
    )
    return flatten_dataset


@partial(jax.jit, static_argnums=(0, 1, 2))
def create_shuffled_path(puzzle: Puzzle, shuffle_length: int, batch_size: int, key: chex.PRNGKey):
    targets = jax.vmap(puzzle.get_target_state)(jax.random.split(key, batch_size))

    def get_trajectory_key(target: Puzzle.State, key: chex.PRNGKey):
        def _scan(carry, _):
            state, before_state, key = carry
            neighbor_states, cost = puzzle.get_neighbours(state, filled=True)
            go_back = jax.vmap(puzzle.is_equal, in_axes=(None, 0))(before_state, neighbor_states)
            filled = jnp.isfinite(cost).astype(jnp.float32)
            filled = jnp.where(go_back, 0.0, filled)
            prob = filled / jnp.sum(filled)
            key, subkey = jax.random.split(key)
            idx = jax.random.choice(subkey, jnp.arange(cost.shape[0]), p=prob)
            next_state = neighbor_states[idx]
            return (next_state, state, key), next_state

        _, moves = jax.lax.scan(_scan, (target, target, key), None, length=shuffle_length)
        return moves
    moves = jax.vmap(get_trajectory_key)(
        targets, jax.random.split(key, batch_size)
    )  # [batch_size, shuffle_length][state...]
    tiled_targets = jax.tree_util.tree_map(
        lambda x: jnp.tile(x[:, jnp.newaxis, ...], (1, shuffle_length) + (x.ndim - 1) * (1,)),
        targets,
    )
    tiled_targets = jax.tree_util.tree_map(
        lambda x: x.reshape((-1, *x.shape[2:])), tiled_targets
    )
    moves = jax.tree_util.tree_map(
        lambda x: x.reshape((-1, *x.shape[2:])), moves
    )
    return tiled_targets, moves
