import math
from functools import partial
from typing import Any, Callable

import chex
import jax
import jax.numpy as jnp
import optax
from puxle import Puzzle

from helpers.sampling import (
    create_hindsight_target_shuffled_path,
    create_hindsight_target_triangular_shuffled_path,
    create_target_shuffled_path,
)
from heuristic.neuralheuristic.spr_neuralheuristic_base import SPRHeuristicModel
from neural_util.target_update import soft_update


def spr_davi_builder(
    minibatch_size: int,
    heuristic_model: SPRHeuristicModel,
    optimizer: optax.GradientTransformation,
    importance_sampling: int = True,
    importance_sampling_alpha: float = 0.5,
    importance_sampling_beta: float = 0.1,
    importance_sampling_eps: float = 1.0,
    spr_loss_weight: float = 1.0,
    ema_tau: float = 0.99,
    n_devices: int = 1,
    use_target_confidence_weighting: bool = False,
):
    def cosine_similarity_loss(p, z):
        z = jax.lax.stop_gradient(z)
        p = p / (jnp.linalg.norm(p, axis=-1, keepdims=True) + 1e-8)
        z = z / (jnp.linalg.norm(z, axis=-1, keepdims=True) + 1e-8)
        return -jnp.mean(jnp.sum(p * z, axis=-1))

    def spr_davi_loss(
        heuristic_params: Any,
        target_heuristic_params: Any,
        preproc: chex.Array,
        next_preproc: chex.Array,
        target_heuristic: chex.Array,
        actions: chex.Array,
        weights: chex.Array,
    ):
        # --- Standard DAVI Loss ---
        (current_heuristic, _, _), variable_updates = heuristic_model.apply(
            heuristic_params, preproc, training=True, mutable=["batch_stats"]
        )
        if n_devices > 1:
            variable_updates = jax.lax.pmean(variable_updates, axis_name="devices")

        new_params = {
            "params": heuristic_params["params"],
            "batch_stats": variable_updates["batch_stats"],
        }
        davi_diff = target_heuristic.squeeze() - current_heuristic.squeeze()
        davi_loss = jnp.mean(jnp.square(davi_diff) * weights)

        # --- SPR Loss ---
        # Online network predictions
        (_, projected_p, pred_next_p_all) = heuristic_model.apply(
            heuristic_params, preproc, training=True, mutable=["batch_stats"]
        )[0]

        pred_next_p_all = jnp.reshape(
            pred_next_p_all, (pred_next_p_all.shape[0], heuristic_model.action_size, -1)
        )
        pred_next_p = jnp.take_along_axis(
            pred_next_p_all, actions[:, jnp.newaxis, jnp.newaxis], axis=1
        ).squeeze(1)

        # Target network predictions
        (_, target_next_p, _) = heuristic_model.apply(
            target_heuristic_params, next_preproc, training=False, mutable=["batch_stats"]
        )[0]

        spr_loss = cosine_similarity_loss(pred_next_p, target_next_p)

        # --- Combined Loss ---
        total_loss = davi_loss + spr_loss_weight * spr_loss

        return total_loss, (new_params, davi_diff, spr_loss)

    def spr_davi(
        key: chex.PRNGKey,
        dataset: dict[str, chex.Array],
        heuristic_params: Any,
        target_heuristic_params: Any,
        opt_state: optax.OptState,
    ):
        """
        SPR-DAVI is a heuristic for the sliding puzzle problem.
        """
        preproc = dataset["preproc"]
        next_preproc = dataset["next_preproc"]
        target_heuristic = dataset["target_heuristic"]
        actions = dataset["actions"]
        diff = dataset["diff"]  # This is davi_diff, used for importance sampling

        data_size = target_heuristic.shape[0]
        batch_size = math.ceil(data_size / minibatch_size)

        if importance_sampling:
            abs_diff = jnp.abs(diff)
            sampling_weights = jnp.power(
                abs_diff + importance_sampling_eps, importance_sampling_alpha
            )
            sampling_probs = sampling_weights / jnp.sum(sampling_weights)
            loss_weights = jnp.power(data_size * sampling_probs, -importance_sampling_beta)
            loss_weights = loss_weights / jnp.max(loss_weights)
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
            )
            loss_weights = jnp.ones_like(batch_indexs)

        if use_target_confidence_weighting:
            cost = dataset["cost"]
            cost_weights = 1.0 / jnp.maximum(cost, 1.0)
            cost_weights = cost_weights / jnp.mean(cost_weights)
            loss_weights = loss_weights * cost_weights
        batch_indexs = jnp.reshape(batch_indexs, (batch_size, minibatch_size))

        batched_preproc = jnp.take(preproc, batch_indexs, axis=0)
        batched_next_preproc = jnp.take(next_preproc, batch_indexs, axis=0)
        batched_target_heuristic = jnp.take(target_heuristic, batch_indexs, axis=0)
        batched_actions = jnp.take(actions, batch_indexs, axis=0)
        batched_weights = jnp.take(loss_weights, batch_indexs, axis=0)

        def train_loop(carry, batched_dataset):
            heuristic_params, target_heuristic_params, opt_state = carry
            preproc, next_preproc, target_h, actions, weights = batched_dataset

            grad_fn = jax.value_and_grad(spr_davi_loss, has_aux=True)
            (loss, (heuristic_params, _, spr_loss)), grads = grad_fn(
                heuristic_params,
                target_heuristic_params,
                preproc,
                next_preproc,
                target_h,
                actions,
                weights,
            )

            if n_devices > 1:
                grads = jax.lax.psum(grads, axis_name="devices")

            updates, opt_state = optimizer.update(grads, opt_state, params=heuristic_params)
            heuristic_params = optax.apply_updates(heuristic_params, updates)

            # EMA update for the target network
            target_heuristic_params = soft_update(
                target_heuristic_params, heuristic_params, ema_tau
            )

            grad_magnitude = jax.tree_util.tree_map(
                lambda x: jnp.abs(jnp.reshape(x, (-1,))), jax.tree_util.tree_leaves(grads["params"])
            )
            grad_magnitude_mean = jnp.mean(jnp.concatenate(grad_magnitude))

            return (heuristic_params, target_heuristic_params, opt_state), (
                loss,
                spr_loss,
                grad_magnitude_mean,
            )

        (heuristic_params, target_heuristic_params, opt_state), (
            losses,
            spr_losses,
            grad_means,
        ) = jax.lax.scan(
            train_loop,
            (heuristic_params, target_heuristic_params, opt_state),
            (
                batched_preproc,
                batched_next_preproc,
                batched_target_heuristic,
                batched_actions,
                batched_weights,
            ),
        )

        loss = jnp.mean(losses)
        spr_loss_mean = jnp.mean(spr_losses)
        grad_magnitude_mean = jnp.mean(grad_means)
        weights_magnitude = jax.tree_util.tree_map(
            lambda x: jnp.abs(jnp.reshape(x, (-1,))),
            jax.tree_util.tree_leaves(heuristic_params["params"]),
        )
        weights_magnitude_mean = jnp.mean(jnp.concatenate(weights_magnitude))

        return (
            heuristic_params,
            target_heuristic_params,
            opt_state,
            loss,
            spr_loss_mean,
            grad_magnitude_mean,
            weights_magnitude_mean,
        )

    if n_devices > 1:

        def pmap_spr_davi(key, dataset, heuristic_params, target_heuristic_params, opt_state):
            keys = jax.random.split(key, n_devices)
            # Assuming dataset is already replicated across devices
            (
                heuristic_params,
                target_heuristic_params,
                opt_state,
                loss,
                spr_loss,
                grad_mag,
                weight_mag,
            ) = jax.pmap(spr_davi, in_axes=(0, 0, None, None, None), axis_name="devices")(
                keys, dataset, heuristic_params, target_heuristic_params, opt_state
            )

            # Sync parameters and states
            heuristic_params = jax.tree_util.tree_map(lambda xs: xs[0], heuristic_params)
            target_heuristic_params = jax.tree_util.tree_map(
                lambda xs: xs[0], target_heuristic_params
            )
            opt_state = jax.tree_util.tree_map(lambda xs: xs[0], opt_state)

            return (
                heuristic_params,
                target_heuristic_params,
                opt_state,
                jnp.mean(loss),
                jnp.mean(spr_loss),
                jnp.mean(grad_mag),
                jnp.mean(weight_mag),
            )

        return pmap_spr_davi
    else:
        return jax.jit(spr_davi)


def _get_datasets(
    puzzle: Puzzle,
    preproc_fn: Callable,
    heuristic_model: SPRHeuristicModel,
    minibatch_size: int,
    target_heuristic_params: Any,
    heuristic_params: Any,
    shuffled_path: dict[str, chex.Array],
    key: chex.PRNGKey,
):
    solve_configs = shuffled_path["solve_configs"]
    states = shuffled_path["states"]
    move_costs = shuffled_path["move_costs"]

    minibatched_solve_configs = jax.tree_util.tree_map(
        lambda x: x.reshape((-1, minibatch_size, *x.shape[1:])), solve_configs
    )
    minibatched_states = jax.tree_util.tree_map(
        lambda x: x.reshape((-1, minibatch_size, *x.shape[1:])), states
    )
    minibatched_move_costs = move_costs.reshape((-1, minibatch_size, *move_costs.shape[1:]))

    def get_minibatched_datasets(_, vals):
        solve_configs, states, move_costs = vals
        solved = puzzle.batched_is_solved(solve_configs, states, multi_solve_config=True)

        # Get neighbors and costs
        neighbors, cost = puzzle.batched_get_neighbours(
            solve_configs, states, filleds=jnp.ones(minibatch_size), multi_solve_config=True
        )

        # --- Target Heuristic Calculation (Standard DAVI) ---
        neighbors_solved = jax.vmap(
            lambda x, y: puzzle.batched_is_solved(x, y, multi_solve_config=True),
            in_axes=(None, 0),
        )(solve_configs, neighbors)
        preproc_neighbors = jax.vmap(jax.vmap(preproc_fn, in_axes=(0, 0)), in_axes=(None, 0))(
            solve_configs, neighbors
        )
        flatten_neighbors = jnp.reshape(
            preproc_neighbors, (-1, minibatch_size, *preproc_neighbors.shape[2:])
        )

        def heur_scan(n_states):
            heur, _, _ = heuristic_model.apply(
                target_heuristic_params, n_states, training=False, mutable=["batch_stats"]
            )[0]
            return heur.squeeze()

        heur = jax.vmap(heur_scan)(flatten_neighbors)
        heur = jnp.maximum(jnp.where(neighbors_solved, 0.0, heur), 0.0)

        target_heuristic_values = heur + cost
        min_indices = jnp.argmin(target_heuristic_values, axis=0)
        target_heuristic = jnp.min(target_heuristic_values, axis=0)
        target_heuristic = jnp.where(solved, 0.0, target_heuristic)

        # --- Data for SPR ---
        # The action is the one that leads to the state with the minimum heuristic
        actions = min_indices

        # The next_state is the one corresponding to the selected action
        batch_size = jax.tree_util.tree_leaves(states)[0].shape[0]
        next_states = jax.tree_util.tree_map(
            lambda x: x[actions, jnp.arange(batch_size), :], neighbors
        )

        # Preprocess current and next states
        preproc = jax.vmap(preproc_fn)(solve_configs, states)
        next_preproc = jax.vmap(preproc_fn)(solve_configs, next_states)

        # --- Diff for Importance Sampling ---
        current_heur, _, _ = heuristic_model.apply(
            heuristic_params, preproc, training=False, mutable=["batch_stats"]
        )[0]
        diff = target_heuristic - current_heur.squeeze()

        return None, (preproc, next_preproc, target_heuristic, actions, diff, move_costs)

    _, (preproc, next_preproc, target_heuristic, actions, diff, cost) = jax.lax.scan(
        get_minibatched_datasets,
        None,
        (minibatched_solve_configs, minibatched_states, minibatched_move_costs),
    )

    preproc = preproc.reshape((-1, *preproc.shape[2:]))
    next_preproc = next_preproc.reshape((-1, *next_preproc.shape[2:]))
    target_heuristic = target_heuristic.reshape((-1, *target_heuristic.shape[2:]))
    actions = actions.reshape((-1, *actions.shape[2:]))
    diff = diff.reshape((-1, *diff.shape[2:]))
    cost = cost.reshape((-1, *cost.shape[2:]))

    return {
        "preproc": preproc,
        "next_preproc": next_preproc,
        "target_heuristic": target_heuristic,
        "actions": actions,
        "diff": diff,
        "cost": cost,
    }


def get_spr_heuristic_dataset_builder(
    puzzle: Puzzle,
    preproc_fn: Callable,
    heuristic_model: SPRHeuristicModel,
    dataset_size: int,
    shuffle_length: int,
    dataset_minibatch_size: int,
    using_hindsight_target: bool = True,
    using_triangular_sampling: bool = False,
    n_devices: int = 1,
):
    if using_hindsight_target:
        shuffle_parallel = int(
            min(math.ceil(dataset_size / shuffle_length), dataset_minibatch_size)
        )
        steps = math.ceil(dataset_size / (shuffle_parallel * shuffle_length))
        if using_triangular_sampling:
            create_shuffled_path_fn = partial(
                create_hindsight_target_triangular_shuffled_path,
                puzzle,
                shuffle_length,
                shuffle_parallel,
            )
        else:
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
        partial(_get_datasets, puzzle, preproc_fn, heuristic_model, dataset_minibatch_size)
    )

    @jax.jit
    def get_datasets(target_heuristic_params: Any, heuristic_params: Any, key: chex.PRNGKey):
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
