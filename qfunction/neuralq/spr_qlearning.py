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
from neural_util.target_update import soft_update
from qfunction.neuralq.qlearning import boltzmann_action_selection
from qfunction.neuralq.spr_neuralq_base import SPRQModel


def spr_qlearning_builder(
    minibatch_size: int,
    q_model: SPRQModel,
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
        return 1.0 - jnp.mean(jnp.sum(p * z, axis=-1))

    def spr_qlearning_loss(
        q_params: Any,
        target_q_params: Any,
        preproc: chex.Array,
        next_preproc: chex.Array,
        actions: chex.Array,
        target_qs: chex.Array,
        weights: chex.Array,
    ):
        # --- Standard Q-Learning Loss ---
        (q_values_at_actions, pred_next_p), variable_updates = q_model.apply(
            q_params,
            preproc,
            actions,
            training=True,
            mutable=["batch_stats"],
            method=q_model.get_q_and_predicted_next_p,
        )
        if n_devices > 1:
            variable_updates = jax.lax.pmean(variable_updates, axis_name="devices")

        new_params = {"params": q_params["params"], "batch_stats": variable_updates["batch_stats"]}
        q_diff = target_qs.squeeze() - q_values_at_actions.squeeze()
        q_loss = jnp.mean(jnp.square(q_diff) * weights)

        # --- SPR Loss ---
        # Target network predictions
        target_next_p = q_model.apply(
            target_q_params,
            next_preproc,
            training=False,
            mutable=["batch_stats"],
            method=q_model.get_projected_p,
        )[0]

        spr_loss = cosine_similarity_loss(pred_next_p, target_next_p)

        # --- Combined Loss ---
        total_loss = q_loss + spr_loss_weight * spr_loss

        return total_loss, (new_params, q_loss, spr_loss)

    def spr_qlearning(
        key: chex.PRNGKey,
        dataset: dict[str, chex.Array],
        q_params: Any,
        target_q_params: Any,
        opt_state: optax.OptState,
    ):
        preproc = dataset["preproc"]
        next_preproc = dataset["next_preproc"]
        target_q = dataset["target_q"]
        actions = dataset["actions"]
        diff = dataset["diff"]  # This is q_diff, used for importance sampling

        data_size = target_q.shape[0]
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
        batched_target_q = jnp.take(target_q, batch_indexs, axis=0)
        batched_actions = jnp.take(actions, batch_indexs, axis=0)
        batched_weights = jnp.take(loss_weights, batch_indexs, axis=0)

        def train_loop(carry, batched_dataset):
            q_params, target_q_params, opt_state = carry
            preproc, next_preproc, target_q, actions, weights = batched_dataset

            grad_fn = jax.value_and_grad(spr_qlearning_loss, has_aux=True)
            (loss, (q_params, q_loss, spr_loss)), grads = grad_fn(
                q_params,
                target_q_params,
                preproc,
                next_preproc,
                actions,
                target_q,
                weights,
            )

            if n_devices > 1:
                grads = jax.lax.psum(grads, axis_name="devices")

            updates, opt_state = optimizer.update(grads, opt_state, params=q_params)
            q_params = optax.apply_updates(q_params, updates)

            target_q_params = soft_update(target_q_params, q_params, ema_tau)

            grad_magnitude = jax.tree_util.tree_map(
                lambda x: jnp.abs(jnp.reshape(x, (-1,))), jax.tree_util.tree_leaves(grads["params"])
            )
            grad_magnitude_mean = jnp.mean(jnp.concatenate(grad_magnitude))

            return (q_params, target_q_params, opt_state), (
                loss,
                q_loss,
                spr_loss,
                grad_magnitude_mean,
            )

        (q_params, target_q_params, opt_state), (
            losses,
            q_losses,
            spr_losses,
            grad_means,
        ) = jax.lax.scan(
            train_loop,
            (q_params, target_q_params, opt_state),
            (
                batched_preproc,
                batched_next_preproc,
                batched_target_q,
                batched_actions,
                batched_weights,
            ),
        )

        loss = jnp.mean(losses)
        q_loss_mean = jnp.mean(q_losses)
        spr_loss_mean = jnp.mean(spr_losses)
        grad_magnitude_mean = jnp.mean(grad_means)
        weights_magnitude = jax.tree_util.tree_map(
            lambda x: jnp.abs(jnp.reshape(x, (-1,))), jax.tree_util.tree_leaves(q_params["params"])
        )
        weights_magnitude_mean = jnp.mean(jnp.concatenate(weights_magnitude))

        return (
            q_params,
            target_q_params,
            opt_state,
            loss,
            q_loss_mean,
            spr_loss_mean,
            grad_magnitude_mean,
            weights_magnitude_mean,
        )

    if n_devices > 1:

        def pmap_spr_qlearning(key, dataset, q_params, target_q_params, opt_state):
            keys = jax.random.split(key, n_devices)
            (
                q_params,
                target_q_params,
                opt_state,
                loss,
                q_loss,
                spr_loss,
                grad_mag,
                weight_mag,
            ) = jax.pmap(spr_qlearning, in_axes=(0, 0, None, None, None), axis_name="devices")(
                keys, dataset, q_params, target_q_params, opt_state
            )

            q_params = jax.tree_util.tree_map(lambda xs: xs[0], q_params)
            target_q_params = jax.tree_util.tree_map(lambda xs: xs[0], target_q_params)
            opt_state = jax.tree_util.tree_map(lambda xs: xs[0], opt_state)

            return (
                q_params,
                target_q_params,
                opt_state,
                jnp.mean(loss),
                jnp.mean(q_loss),
                jnp.mean(spr_loss),
                jnp.mean(grad_mag),
                jnp.mean(weight_mag),
            )

        return pmap_spr_qlearning
    else:
        return jax.jit(spr_qlearning)


def _get_datasets_with_policy(
    puzzle: Puzzle,
    preproc_fn: Callable,
    q_model: SPRQModel,
    minibatch_size: int,
    target_q_params: Any,
    q_params: Any,
    shuffled_path: dict[str, chex.Array],
    key: chex.PRNGKey,
    temperature: float = 1.0 / 3.0,
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

    def get_minibatched_datasets(key, vals):
        key, subkey = jax.random.split(key)
        solve_configs, states, move_costs = vals
        solved = puzzle.batched_is_solved(solve_configs, states, multi_solve_config=True)

        preproc = jax.vmap(preproc_fn)(solve_configs, states)
        q_values = q_model.apply(
            q_params, preproc, training=False, mutable=["batch_stats"], method=q_model.get_q
        )[0]

        neighbors, cost = puzzle.batched_get_neighbours(
            solve_configs, states, filleds=jnp.ones(minibatch_size), multi_solve_config=True
        )
        mask = jnp.isfinite(jnp.transpose(cost, (1, 0)))

        # Action selection
        probs = boltzmann_action_selection(q_values, temperature=temperature, mask=mask)
        idxs = jnp.arange(q_values.shape[1])
        actions = jax.vmap(lambda k, p: jax.random.choice(k, idxs, p=p))(
            jax.random.split(subkey, q_values.shape[0]), probs
        )

        # Get next state based on action
        batch_size = actions.shape[0]
        selected_neighbors = jax.tree_util.tree_map(
            lambda x: x[actions, jnp.arange(batch_size), :], neighbors
        )

        # --- Target Q Calculation ---
        _, neighbor_cost = puzzle.batched_get_neighbours(
            solve_configs,
            selected_neighbors,
            filleds=jnp.ones(minibatch_size),
            multi_solve_config=True,
        )
        selected_neighbors_solved = puzzle.batched_is_solved(
            solve_configs, selected_neighbors, multi_solve_config=True
        )

        next_preproc = jax.vmap(preproc_fn)(solve_configs, selected_neighbors)

        # Use target network for next state Q-values
        next_q_values = q_model.apply(
            target_q_params,
            next_preproc,
            training=False,
            mutable=["batch_stats"],
            method=q_model.get_q,
        )[0]

        mask_neighbor = jnp.isfinite(jnp.transpose(neighbor_cost, (1, 0)))
        next_q_values = jnp.where(mask_neighbor, next_q_values, jnp.inf)
        argmin_q = jnp.argmin(next_q_values, axis=1)
        min_q = jnp.take_along_axis(next_q_values, argmin_q[:, jnp.newaxis], axis=1).squeeze(1)
        selected_neighbor_costs = jnp.take_along_axis(
            neighbor_cost, argmin_q[jnp.newaxis, :], axis=0
        ).squeeze(0)
        target_q = jnp.maximum(min_q, 0.0) + selected_neighbor_costs
        target_q = jnp.where(selected_neighbors_solved, 0, target_q)
        target_q = jnp.where(solved, 0.0, target_q)

        # --- Diff for Importance Sampling ---
        q_values_at_actions = jnp.take_along_axis(
            q_values, actions[:, jnp.newaxis], axis=1
        ).squeeze(1)
        diff = target_q - q_values_at_actions

        return key, (preproc, next_preproc, target_q, actions, diff, move_costs)

    _, (preproc, next_preproc, target_q, actions, diff, cost) = jax.lax.scan(
        get_minibatched_datasets,
        key,
        (minibatched_solve_configs, minibatched_states, minibatched_move_costs),
    )

    preproc = preproc.reshape((-1, *preproc.shape[2:]))
    next_preproc = next_preproc.reshape((-1, *next_preproc.shape[2:]))
    target_q = target_q.reshape((-1, *target_q.shape[2:]))
    actions = actions.reshape((-1, *actions.shape[2:]))
    diff = diff.reshape((-1, *diff.shape[2:]))
    cost = cost.reshape((-1, *cost.shape[2:]))

    return {
        "preproc": preproc,
        "next_preproc": next_preproc,
        "target_q": target_q,
        "actions": actions,
        "diff": diff,
        "cost": cost,
    }


def get_spr_qlearning_dataset_builder(
    puzzle: Puzzle,
    preproc_fn: Callable,
    q_model: SPRQModel,
    dataset_size: int,
    shuffle_length: int,
    dataset_minibatch_size: int,
    using_hindsight_target: bool = True,
    using_triangular_sampling: bool = False,
    with_policy: bool = True,  # Q-learning usually needs a policy
    n_devices: int = 1,
    temperature: float = 1.0 / 3.0,
):
    if using_hindsight_target:
        assert not puzzle.fixed_target, "Fixed target not supported for hindsight target"
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

    # For now, we only implement the policy-based version
    jited_get_datasets = jax.jit(
        partial(
            _get_datasets_with_policy,
            puzzle,
            preproc_fn,
            q_model,
            dataset_minibatch_size,
            temperature=temperature,
        )
    )

    @jax.jit
    def get_datasets(target_q_params: Any, q_params: Any, key: chex.PRNGKey):
        def scan_fn(key, _):
            key, subkey = jax.random.split(key)
            paths = create_shuffled_path_fn(subkey)
            return key, paths

        key, paths = jax.lax.scan(scan_fn, key, None, length=steps)
        paths = jax.tree_util.tree_map(
            lambda x: x.reshape((-1, *x.shape[2:]))[:dataset_size], paths
        )

        flatten_dataset = jited_get_datasets(target_q_params, q_params, paths, key)
        return flatten_dataset

    if n_devices > 1:

        def pmap_get_datasets(target_q_params, q_params, key):
            keys = jax.random.split(key, n_devices)
            datasets = jax.pmap(get_datasets, in_axes=(None, None, 0))(
                target_q_params, q_params, keys
            )
            return datasets

        return pmap_get_datasets
    else:
        return get_datasets
