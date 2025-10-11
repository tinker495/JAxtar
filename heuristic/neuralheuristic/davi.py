import math
from functools import partial
from typing import Any, Callable, Optional

import chex
import jax
import jax.numpy as jnp
import optax
import xtructure.numpy as xnp
from puxle import Puzzle

from heuristic.neuralheuristic.neuralheuristic_base import HeuristicBase
from train_util.losses import loss_from_diff
from train_util.sampling import (
    create_hindsight_target_shuffled_path,
    create_hindsight_target_triangular_shuffled_path,
    create_target_shuffled_path,
)
from train_util.util import (
    apply_with_conditional_batch_stats,
    build_new_params_from_updates,
)


def davi_builder(
    minibatch_size: int,
    heuristic_model: HeuristicBase,
    optimizer: optax.GradientTransformation,
    preproc_fn: Callable,
    n_devices: int = 1,
    use_target_confidence_weighting: bool = False,
    use_target_sharpness_weighting: bool = False,
    target_sharpness_alpha: float = 1.0,
    using_priority_sampling: bool = False,
    per_alpha: float = 0.6,
    per_beta: float = 0.4,
    per_epsilon: float = 1e-6,
    loss_type: str = "mse",
    huber_delta: float = 0.1,
    replay_ratio: int = 1,
    td_error_clip: Optional[float] = None,
):
    def davi_loss(
        heuristic_params: Any,
        solveconfigs: chex.Array,
        states: chex.Array,
        target_heuristic: chex.Array,
        weights: chex.Array,
    ):
        # Preprocess during training
        preproc = jax.vmap(preproc_fn)(solveconfigs, states)
        current_heuristic, variable_updates = apply_with_conditional_batch_stats(
            heuristic_model.apply, heuristic_params, preproc, training=True, n_devices=n_devices
        )
        new_params = build_new_params_from_updates(heuristic_params, variable_updates)
        diff = target_heuristic.squeeze() - current_heuristic.squeeze()
        if td_error_clip is not None and td_error_clip > 0:
            clip_val = jnp.asarray(td_error_clip, dtype=diff.dtype)
            diff = jnp.clip(diff, -clip_val, clip_val)
        per_sample = loss_from_diff(diff, loss=loss_type, huber_delta=huber_delta)
        loss_value = jnp.mean(per_sample * weights)
        return loss_value, (new_params, diff)

    def davi(
        key: chex.PRNGKey,
        dataset: dict[str, chex.Array],
        heuristic_params: Any,
        opt_state: optax.OptState,
    ):
        """
        DAVI is a heuristic for the sliding puzzle problem.
        """
        solveconfigs = dataset["solveconfigs"]
        states = dataset["states"]
        target_heuristic = dataset["target_heuristic"]
        data_size = target_heuristic.shape[0]
        batch_size = math.ceil(data_size / minibatch_size)

        if using_priority_sampling:
            diff = dataset["diff"]
            # diff is already clipped in _get_datasets if td_error_clip is enabled
            # Sanitize TD errors to avoid NaN/Inf poisoning
            diff = jnp.nan_to_num(diff, nan=0.0, posinf=1e6, neginf=-1e6)

            # Calculate priorities based on TD error with strict positivity
            priorities = jnp.abs(diff) + per_epsilon
            priorities = jnp.clip(priorities, a_min=1e-12)

            # Stable sampling probabilities in log-space: p_i ∝ priorities^alpha
            logp = per_alpha * jnp.log(priorities)
            logp = logp - jnp.max(logp)
            sampling_probs = jnp.exp(logp)
            sampling_probs = sampling_probs / (jnp.sum(sampling_probs) + 1e-12)

            # Stable importance sampling weights in log-space; max-normalized to 1
            clipped_probs = jnp.clip(sampling_probs, a_min=1e-12)
            log_w = -per_beta * (jnp.log(data_size) + jnp.log(clipped_probs))
            log_w = log_w - jnp.max(log_w)
            is_weights = jnp.exp(log_w)
            loss_weights = is_weights
        else:
            loss_weights = jnp.ones(data_size)

        if use_target_confidence_weighting:
            cost = dataset["cost"]
            cost_weights = 1.0 / jnp.sqrt(jnp.maximum(cost, 1.0))
            cost_weights = cost_weights / jnp.mean(cost_weights)
            loss_weights = loss_weights * cost_weights

        if use_target_sharpness_weighting and ("target_entropy" in dataset):
            entropy = dataset["target_entropy"]
            max_entropy = dataset.get("target_entropy_max", None)
            if max_entropy is not None:
                normalized_entropy = entropy / (max_entropy + 1e-8)
                sharpness = 1.0 - normalized_entropy
                sharp_weights = 1.0 + target_sharpness_alpha * sharpness
                sharp_weights = sharp_weights / (jnp.mean(sharp_weights) + 1e-8)
                loss_weights = loss_weights * sharp_weights

        if not using_priority_sampling:
            loss_weights = loss_weights / jnp.mean(loss_weights)

        def train_loop(carry, batched_dataset):
            heuristic_params, opt_state = carry
            solveconfigs, states, target_heuristic, weights = batched_dataset
            (loss, (heuristic_params, diff)), grads = jax.value_and_grad(davi_loss, has_aux=True)(
                heuristic_params,
                solveconfigs,
                states,
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
            return (heuristic_params, opt_state), (loss, grad_magnitude_mean, diff)

        # Repeat training loop for replay_ratio times with reshuffling
        def replay_loop(carry, replay_key):
            heuristic_params, opt_state = carry

            # Reshuffle the batch indices for each replay iteration
            if using_priority_sampling:
                # For priority sampling, resample based on priorities
                batch_indexs_replay = jax.random.choice(
                    replay_key,
                    jnp.arange(data_size),
                    shape=(batch_size * minibatch_size,),
                    p=sampling_probs,
                    replace=True,
                )
                loss_weights_replay = is_weights
            else:
                # For uniform sampling, create new permutation
                key_perm_replay, key_fill_replay = jax.random.split(replay_key)
                batch_indexs_replay = jnp.concatenate(
                    [
                        jax.random.permutation(key_perm_replay, jnp.arange(data_size)),
                        jax.random.randint(
                            key_fill_replay,
                            (batch_size * minibatch_size - data_size,),
                            0,
                            data_size,
                        ),
                    ],
                    axis=0,
                )
                loss_weights_replay = loss_weights

            batch_indexs_replay = jnp.reshape(batch_indexs_replay, (batch_size, minibatch_size))

            # Create new batches with reshuffled indices
            batched_solveconfigs_replay = xnp.take(solveconfigs, batch_indexs_replay, axis=0)
            batched_states_replay = xnp.take(states, batch_indexs_replay, axis=0)
            batched_target_heuristic_replay = jnp.take(
                target_heuristic, batch_indexs_replay, axis=0
            )
            batched_weights_replay = jnp.take(loss_weights_replay, batch_indexs_replay, axis=0)
            # Normalize weights per batch to prevent scale drift
            batched_weights_replay = batched_weights_replay / (
                jnp.mean(batched_weights_replay, axis=1, keepdims=True) + 1e-8
            )

            (heuristic_params, opt_state), (losses, grad_magnitude_means, diffs) = jax.lax.scan(
                train_loop,
                (heuristic_params, opt_state),
                (
                    batched_solveconfigs_replay,
                    batched_states_replay,
                    batched_target_heuristic_replay,
                    batched_weights_replay,
                ),
            )
            return (heuristic_params, opt_state), (losses, grad_magnitude_means, diffs)

        # Generate separate keys for each replay iteration
        replay_keys = jax.random.split(key, replay_ratio)
        (heuristic_params, opt_state), (losses, grad_magnitude_means, diffs) = jax.lax.scan(
            replay_loop,
            (heuristic_params, opt_state),
            replay_keys,
        )
        loss = jnp.mean(losses)
        diffs = diffs.reshape(-1)
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
            diffs,
        )

    if n_devices > 1:

        def pmap_davi(key, dataset, heuristic_params, opt_state):
            keys = jax.random.split(key, n_devices)
            (heuristic_params, opt_state, loss, grad_magnitude, weight_magnitude, diffs) = jax.pmap(
                davi, in_axes=(0, 0, None, None), axis_name="devices"
            )(keys, dataset, heuristic_params, opt_state)
            heuristic_params = jax.tree_util.tree_map(lambda xs: xs[0], heuristic_params)
            opt_state = jax.tree_util.tree_map(lambda xs: xs[0], opt_state)
            loss = jnp.mean(loss)
            grad_magnitude = jnp.mean(grad_magnitude)
            weight_magnitude = jnp.mean(weight_magnitude)
            diffs = diffs.reshape(-1)
            return heuristic_params, opt_state, loss, grad_magnitude, weight_magnitude, diffs

        return pmap_davi
    else:
        return jax.jit(davi)


def _get_datasets(
    puzzle: Puzzle,
    preproc_fn: Callable,
    heuristic_model: HeuristicBase,
    minibatch_size: int,
    target_heuristic_params: Any,
    heuristic_params: Any,
    shuffled_path: dict[str, chex.Array],
    key: chex.PRNGKey,
    temperature: float = 1.0 / 3.0,
    td_error_clip: Optional[float] = None,
    use_diffusion_distance: bool = False,
):
    solve_configs = shuffled_path["solve_configs"]
    states = shuffled_path["states"]
    move_costs = shuffled_path["move_costs"]

    minibatched_solve_configs = solve_configs.reshape((-1, minibatch_size))
    minibatched_states = states.reshape((-1, minibatch_size))
    minibatched_move_costs = move_costs.reshape((-1, minibatch_size))

    def get_minibatched_datasets(_, vals):
        solve_configs, states, move_costs = vals
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
            heur = heuristic_model.apply(target_heuristic_params, neighbors, training=False)
            return heur.squeeze()

        heur = jax.vmap(heur_scan)(flatten_neighbors)  # [action_size, batch_size]
        heur = jnp.maximum(jnp.where(neighbors_solved, 0.0, heur), 0.0)
        backup = heur + cost  # [action_size, batch_size]
        target_heuristic = jnp.min(backup, axis=0)
        target_heuristic = jnp.where(
            solved, 0.0, target_heuristic
        )  # if the puzzle is already solved, the heuristic is 0

        if use_diffusion_distance:
            diffusion_targets = jnp.where(solved, 0.0, move_costs)
            target_heuristic = diffusion_targets

        # Target entropy over next-state backup distribution
        safe_temperature = jnp.maximum(temperature, 1e-8)
        backup_bt = jnp.transpose(backup, (1, 0))  # [batch_size, action_size]
        scaled_next = -backup_bt / safe_temperature
        next_probs = jax.nn.softmax(scaled_next, axis=1)
        next_probs = next_probs / (jnp.sum(next_probs, axis=1, keepdims=True) + 1e-8)
        target_entropy = -jnp.sum(
            next_probs * jnp.log(jnp.clip(next_probs, a_min=1e-12)), axis=1
        )  # [batch_size]
        # For solved states or if any neighbor is solved, entropy should be near zero
        any_neighbor_solved = jnp.any(neighbors_solved, axis=0)
        target_entropy = jnp.where(jnp.logical_or(solved, any_neighbor_solved), 0.0, target_entropy)
        # Maximum entropy per state based on action size
        action_size = backup_bt.shape[1]
        max_ent_val = jnp.log(jnp.maximum(jnp.array(action_size, dtype=next_probs.dtype), 1.0))
        target_entropy_max = jnp.full((next_probs.shape[0],), max_ent_val)

        preproc = jax.vmap(preproc_fn)(solve_configs, states)
        heur = heuristic_model.apply(heuristic_params, preproc, training=False)
        diff = target_heuristic - heur.squeeze()
        if td_error_clip is not None and td_error_clip > 0:
            clip_val = jnp.asarray(td_error_clip, dtype=diff.dtype)
            diff = jnp.clip(diff, -clip_val, clip_val)
        return None, (
            solve_configs,
            states,
            target_heuristic,
            diff,
            target_entropy,
            target_entropy_max,
            move_costs,
        )

    _, (
        solve_configs,
        states,
        target_heuristic,
        diff,
        target_entropy,
        target_entropy_max,
        cost,
    ) = jax.lax.scan(
        get_minibatched_datasets,
        None,
        (minibatched_solve_configs, minibatched_states, minibatched_move_costs),
    )

    solve_configs = solve_configs.reshape((-1,))
    states = states.reshape((-1,))
    target_heuristic = target_heuristic.reshape((-1,))
    diff = diff.reshape((-1,))
    target_entropy = target_entropy.reshape((-1,))
    target_entropy_max = target_entropy_max.reshape((-1,))
    cost = cost.reshape((-1,))

    return {
        "solveconfigs": solve_configs,
        "states": states,
        "target_heuristic": target_heuristic,
        "diff": diff,
        "target_entropy": target_entropy,
        "target_entropy_max": target_entropy_max,
        "cost": cost,
    }


def get_heuristic_dataset_builder(
    puzzle: Puzzle,
    preproc_fn: Callable,
    heuristic_model: HeuristicBase,
    dataset_size: int,
    shuffle_length: int,
    dataset_minibatch_size: int,
    using_hindsight_target: bool = True,
    using_triangular_sampling: bool = False,
    n_devices: int = 1,
    temperature: float = 1.0 / 3.0,
    td_error_clip: Optional[float] = None,
    use_diffusion_distance: bool = False,
):

    if using_hindsight_target:
        # Calculate appropriate shuffle_parallel for hindsight sampling
        # For hindsight, we're sampling from lower triangle with (L*(L+1))/2 elements
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
                True,
            )
        else:
            create_shuffled_path_fn = partial(
                create_hindsight_target_shuffled_path,
                puzzle,
                shuffle_length,
                shuffle_parallel,
                True,
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
            True,
        )

    jited_create_shuffled_path = jax.jit(create_shuffled_path_fn)

    jited_get_datasets = jax.jit(
        partial(
            _get_datasets,
            puzzle,
            preproc_fn,
            heuristic_model,
            dataset_minibatch_size,
            temperature=temperature,
            td_error_clip=td_error_clip,
            use_diffusion_distance=use_diffusion_distance,
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
        for k, v in paths.items():
            paths[k] = v.flatten()[:dataset_size]

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
