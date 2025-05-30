import math
from functools import partial
from typing import Any, Callable

import chex
import jax
import jax.numpy as jnp
import optax

from helpers.sampling import (
    create_hindsight_target_shuffled_path,
    create_hindsight_target_triangular_shuffled_path,
    create_target_shuffled_path,
)
from heuristic.neuralheuristic.neuralheuristic_base import NeuralHeuristicBase
from puzzle.puzzle_base import Puzzle


def davi_builder(
    minibatch_size: int,
    heuristic_model: NeuralHeuristicBase,
    optimizer: optax.GradientTransformation,
    importance_sampling: int = True,
    importance_sampling_alpha: float = 0.5,
    importance_sampling_beta: float = 0.1,
    importance_sampling_eps: float = 1.0,
    n_devices: int = 1,
):
    def davi_loss(
        heuristic_params: Any,
        preproc: chex.Array,
        target_heuristic: chex.Array,
        weights: chex.Array,
    ):
        current_heuristic, variable_updates = heuristic_model.apply(
            heuristic_params, preproc, training=True, mutable=["batch_stats"]
        )
        if n_devices > 1:
            variable_updates = jax.lax.pmean(variable_updates, axis_name="devices")
        new_params = {
            "params": heuristic_params["params"],
            "batch_stats": variable_updates["batch_stats"],
        }
        diff = target_heuristic.squeeze() - current_heuristic.squeeze()
        # loss = jnp.mean(hubberloss(diff, delta=0.1) / 0.1 * weights)
        loss = jnp.mean(jnp.square(diff) * weights)
        return loss, new_params

    def davi(
        key: chex.PRNGKey,
        dataset: dict[str, chex.Array],
        heuristic_params: Any,
        opt_state: optax.OptState,
    ):
        """
        DAVI is a heuristic for the sliding puzzle problem.
        """
        preproc = dataset["preproc"]
        target_heuristic = dataset["target_heuristic"]
        diff = dataset["diff"]
        data_size = target_heuristic.shape[0]
        batch_size = math.ceil(data_size / minibatch_size)

        if importance_sampling:
            # Calculate sampling probabilities based on diff (error) for importance sampling
            # Higher diff values get higher probability (similar to PER - Prioritized Experience Replay)
            abs_diff = jnp.abs(diff)
            sampling_weights = jnp.power(
                abs_diff + importance_sampling_eps, importance_sampling_alpha
            )
            sampling_probs = sampling_weights / jnp.sum(sampling_weights)
            loss_weights = jnp.power(data_size * sampling_probs, -importance_sampling_beta)
            loss_weights = loss_weights / jnp.max(loss_weights)

            # Sample indices based on the calculated probabilities
            batch_indexs = jax.random.choice(
                key,
                jnp.arange(data_size),
                shape=(batch_size * minibatch_size,),
                replace=False,
                p=sampling_probs,
            )
        else:
            batch_indexs = jnp.concatenate(
                [
                    jax.random.permutation(key, jnp.arange(data_size)),
                    jax.random.randint(
                        key, (batch_size * minibatch_size - data_size,), 0, data_size
                    ),
                ],
                axis=0,
            )  # [batch_size * minibatch_size]
            loss_weights = jnp.ones_like(batch_indexs)
        batch_indexs = jnp.reshape(batch_indexs, (batch_size, minibatch_size))

        batched_preproc = jnp.take(preproc, batch_indexs, axis=0)
        batched_target_heuristic = jnp.take(target_heuristic, batch_indexs, axis=0)
        batched_weights = jnp.take(loss_weights, batch_indexs, axis=0)

        def train_loop(carry, batched_dataset):
            heuristic_params, opt_state = carry
            preproc, target_heuristic, weights = batched_dataset
            (loss, heuristic_params), grads = jax.value_and_grad(davi_loss, has_aux=True)(
                heuristic_params,
                preproc,
                target_heuristic,
                weights,
            )
            if n_devices > 1:
                grads = jax.lax.psum(grads, axis_name="devices")
            updates, opt_state = optimizer.update(grads, opt_state, params=heuristic_params)
            heuristic_params = optax.apply_updates(heuristic_params, updates)
            # Calculate gradient magnitude mean
            grad_magnitude = jax.tree_util.tree_map(
                lambda x: jnp.abs(jnp.reshape(x, (-1,))), jax.tree_util.tree_leaves(grads["params"])
            )
            grad_magnitude_mean = jnp.mean(jnp.concatenate(grad_magnitude))
            return (heuristic_params, opt_state), (loss, grad_magnitude_mean)

        (heuristic_params, opt_state), (losses, grad_magnitude_means) = jax.lax.scan(
            train_loop,
            (heuristic_params, opt_state),
            (batched_preproc, batched_target_heuristic, batched_weights),
        )
        loss = jnp.mean(losses)
        # Calculate weights magnitude means
        grad_magnitude_mean = jnp.mean(grad_magnitude_means)
        weights_magnitude = jax.tree_util.tree_map(
            lambda x: jnp.abs(jnp.reshape(x, (-1,))),
            jax.tree_util.tree_leaves(heuristic_params["params"]),
        )
        weights_magnitude_mean = jnp.mean(jnp.concatenate(weights_magnitude))
        return (
            heuristic_params,
            opt_state,
            loss,
            grad_magnitude_mean,
            weights_magnitude_mean,
        )

    if n_devices > 1:

        def pmap_davi(key, dataset, heuristic_params, opt_state):
            keys = jax.random.split(key, n_devices)
            (heuristic_params, opt_state, loss, grad_magnitude, weight_magnitude,) = jax.pmap(
                davi, in_axes=(0, 0, None, None), axis_name="devices"
            )(keys, dataset, heuristic_params, opt_state)
            heuristic_params = jax.tree_util.tree_map(lambda xs: xs[0], heuristic_params)
            opt_state = jax.tree_util.tree_map(lambda xs: xs[0], opt_state)
            loss = jnp.mean(loss)
            grad_magnitude = jnp.mean(grad_magnitude)
            weight_magnitude = jnp.mean(weight_magnitude)
            return heuristic_params, opt_state, loss, grad_magnitude, weight_magnitude

        return pmap_davi
    else:
        return jax.jit(davi)


def _get_datasets(
    puzzle: Puzzle,
    preproc_fn: Callable,
    heuristic_model: NeuralHeuristicBase,
    minibatch_size: int,
    target_heuristic_params: Any,
    heuristic_params: Any,
    shuffled_path: dict[str, chex.Array],
    key: chex.PRNGKey,
):
    solve_configs = shuffled_path["solve_configs"]
    states = shuffled_path["states"]

    minibatched_solve_configs = jax.tree_util.tree_map(
        lambda x: x.reshape((-1, minibatch_size, *x.shape[1:])), solve_configs
    )
    minibatched_states = jax.tree_util.tree_map(
        lambda x: x.reshape((-1, minibatch_size, *x.shape[1:])), states
    )

    def get_minibatched_datasets(_, vals):
        solve_configs, states = vals
        solved = puzzle.batched_is_solved(
            solve_configs, states, multi_solve_config=True
        )  # [batch_size]
        neighbors, cost = puzzle.batched_get_neighbours(
            solve_configs, states, filleds=jnp.ones(minibatch_size), multi_solve_config=True
        )  # [action_size, batch_size] [action_size, batch_size]
        neighbors_solved = jax.vmap(
            lambda x, y: puzzle.batched_is_solved(x, y, multi_solve_config=True),
            in_axes=(None, 0),
        )(
            solve_configs, neighbors
        )  # [action_size, batch_size]
        preproc_neighbors = jax.vmap(jax.vmap(preproc_fn, in_axes=(0, 0)), in_axes=(None, 0))(
            solve_configs, neighbors
        )
        # preproc_neighbors: [action_size, batch_size, ...]
        flatten_neighbors = jnp.reshape(
            preproc_neighbors, (-1, minibatch_size, *preproc_neighbors.shape[2:])
        )

        def heur_scan(neighbors):
            heur, _ = heuristic_model.apply(
                target_heuristic_params, neighbors, training=False, mutable=["batch_stats"]
            )
            return heur.squeeze()

        heur = jax.vmap(heur_scan)(flatten_neighbors)  # [action_size, batch_size]
        heur = jnp.maximum(jnp.where(neighbors_solved, 0.0, heur), 0.0)
        target_heuristic = jnp.min(heur + cost, axis=0)
        target_heuristic = jnp.where(
            solved, 0.0, target_heuristic
        )  # if the puzzle is already solved, the heuristic is 0

        preproc = jax.vmap(preproc_fn)(solve_configs, states)
        heur, _ = heuristic_model.apply(
            heuristic_params, preproc, training=False, mutable=["batch_stats"]
        )
        diff = target_heuristic - heur.squeeze()
        return None, (preproc, target_heuristic, diff)

    _, (preproc, target_heuristic, diff) = jax.lax.scan(
        get_minibatched_datasets,
        None,
        (minibatched_solve_configs, minibatched_states),
    )

    preproc = preproc.reshape((-1, *preproc.shape[2:]))
    target_heuristic = target_heuristic.reshape((-1, *target_heuristic.shape[2:]))
    diff = diff.reshape((-1, *diff.shape[2:]))

    return {
        "preproc": preproc,
        "target_heuristic": target_heuristic,
        "diff": diff,
    }


def get_heuristic_dataset_builder(
    puzzle: Puzzle,
    preproc_fn: Callable,
    heuristic_model: NeuralHeuristicBase,
    dataset_size: int,
    shuffle_length: int,
    dataset_minibatch_size: int,
    using_hindsight_target: bool = True,
    using_triangular_target: bool = False,
    n_devices: int = 1,
):

    if using_hindsight_target:
        # Calculate appropriate shuffle_parallel for hindsight sampling
        # For hindsight, we're sampling from lower triangle with (L*(L+1))/2 elements
        if using_triangular_target:
            triangle_size = shuffle_length * (shuffle_length + 1) // 2
            needed_parallel = math.ceil(dataset_size / triangle_size)
            shuffle_parallel = int(min(needed_parallel, dataset_minibatch_size))
            steps = math.ceil(dataset_size / (shuffle_parallel * triangle_size))
            create_shuffled_path_fn = partial(
                create_hindsight_target_triangular_shuffled_path,
                puzzle,
                shuffle_length,
                shuffle_parallel,
            )
        else:
            shuffle_parallel = int(
                min(math.ceil(dataset_size / shuffle_length), dataset_minibatch_size)
            )
            steps = math.ceil(dataset_size / (shuffle_parallel * shuffle_length))
            create_shuffled_path_fn = partial(
                create_hindsight_target_shuffled_path,
                puzzle,
                shuffle_length,
                shuffle_parallel,
            )
    else:
        shuffle_parallel = int(
            min(math.ceil(dataset_size / shuffle_length), dataset_minibatch_size)
        )
        steps = math.ceil(dataset_size / (shuffle_parallel * shuffle_length))
        create_shuffled_path_fn = partial(
            create_target_shuffled_path,
            puzzle,
            shuffle_length,
            shuffle_parallel,
        )

    jited_create_shuffled_path = jax.jit(create_shuffled_path_fn)

    jited_get_datasets = jax.jit(
        partial(
            _get_datasets,
            puzzle,
            preproc_fn,
            heuristic_model,
            dataset_minibatch_size,
        )
    )

    @jax.jit
    def get_datasets(
        target_heuristic_params: Any,
        heuristic_params: Any,
        key: chex.PRNGKey,
    ):
        def scan_fn(key, _):
            key, subkey = jax.random.split(key)
            paths = jited_create_shuffled_path(subkey)
            return key, paths

        key, paths = jax.lax.scan(scan_fn, key, None, length=steps)
        paths = jax.tree_util.tree_map(
            lambda x: x.reshape((-1, *x.shape[2:]))[:dataset_size], paths
        )

        flatten_dataset = jited_get_datasets(target_heuristic_params, heuristic_params, paths, key)
        return flatten_dataset

    if n_devices > 1:

        def pmap_get_datasets(target_heuristic_params, heuristic_params, key):
            keys = jax.random.split(key, n_devices)
            datasets = jax.pmap(get_datasets, in_axes=(None, None, 0))(
                target_heuristic_params, heuristic_params, keys
            )
            return datasets

        return pmap_get_datasets
    else:
        return get_datasets
