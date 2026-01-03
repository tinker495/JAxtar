import math
from typing import Any, Callable, Optional

import chex
import jax
import jax.numpy as jnp
import optax

from neural_util.basemodel import DistanceHLGModel, DistanceModel
from train_util.sampling import minibatch_datasets
from train_util.util import (
    apply_with_conditional_batch_stats,
    build_new_params_from_updates,
    get_self_predictive_train_args,
)


def qfunction_train_builder(
    minibatch_size: int,
    q_fn: DistanceModel | DistanceHLGModel,
    optimizer: optax.GradientTransformation,
    preproc_fn: Callable,
    n_devices: int = 1,
    loss_type: str = "mse",
    loss_args: Optional[dict[str, Any]] = None,
    replay_ratio: int = 1,
):
    def qfunction_train_loss(
        q_params: Any,
        preproc: chex.Array,
        actions: chex.Array,
        target_qs: chex.Array,
        path_actions: chex.Array,
        ema_latents: chex.Array,
        same_trajectory_masks: chex.Array,
        weights: chex.Array,
    ):
        # Preprocess during training
        (per_sample_loss, aux), variable_updates = apply_with_conditional_batch_stats(
            q_fn.apply,
            q_params,
            preproc,
            target_qs,
            actions,
            training=True,
            n_devices=n_devices,
            method=q_fn.train_loss,
            loss_type=loss_type,
            loss_args=loss_args,
            path_actions=path_actions,
            ema_latents=ema_latents,
            same_trajectory_masks=same_trajectory_masks,
        )
        new_params = build_new_params_from_updates(q_params, variable_updates)
        loss_value = jnp.mean(per_sample_loss.squeeze() * weights)
        return loss_value, (new_params, aux)

    def qfunction_train(
        key: chex.PRNGKey,
        dataset: dict[str, chex.Array],
        q_params: Any,
        target_q_params: Any,
        opt_state: optax.OptState,
    ):
        """Run one optimization epoch of neural Q-learning for the provided puzzle dataset."""
        solveconfigs = dataset["solveconfigs"]
        states = dataset["states"]
        target_q = dataset["target_q"]
        actions = dataset["actions"]
        path_actions = dataset["path_actions"]
        trajectory_indices = dataset["trajectory_indices"]
        step_indices = dataset["step_indices"]
        data_size = target_q.shape[0]
        batch_size = math.ceil(data_size / minibatch_size)

        loss_weights = jnp.ones(data_size)
        loss_weights = loss_weights / jnp.mean(loss_weights)

        def train_loop(carry, batched_dataset):
            q_params, opt_state = carry
            (
                solveconfigs,
                states,
                target_q,
                actions,
                path_actions,
                trajectory_indices,
                step_indices,
                weights,
            ) = batched_dataset

            preprocessed_states = jax.vmap(jax.vmap(preproc_fn))(solveconfigs, states)
            (
                ema_next_state_latents,
                path_actions,
                same_trajectory_masks,
            ) = get_self_predictive_train_args(
                q_fn,
                target_q_params,
                preprocessed_states,
                path_actions,
                trajectory_indices,
                step_indices,
            )

            (loss, (q_params, aux)), grads = jax.value_and_grad(qfunction_train_loss, has_aux=True)(
                q_params,
                preprocessed_states,
                actions,
                target_q,
                path_actions,
                ema_next_state_latents,
                same_trajectory_masks,
                weights,
            )
            if n_devices > 1:
                grads = jax.lax.psum(grads, axis_name="devices")
            updates, opt_state = optimizer.update(grads, opt_state, params=q_params)
            q_params = optax.apply_updates(q_params, updates)
            return (q_params, opt_state), (loss, aux)

        def replay_loop(carry, replay_key):
            q_params, opt_state = carry

            (
                batched_solveconfigs,
                batched_states,
                batched_target_q,
                batched_actions,
                batched_path_actions,
                batched_trajectory_indices,
                batched_step_indices,
                batched_weights,
            ) = minibatch_datasets(
                solveconfigs,
                states,
                target_q,
                actions,
                path_actions,
                trajectory_indices,
                step_indices,
                loss_weights,
                data_size=data_size,
                batch_size=batch_size,
                minibatch_size=minibatch_size,
                key=replay_key,
            )

            (q_params, opt_state,), (losses, auxs) = jax.lax.scan(
                train_loop,
                (q_params, opt_state),
                (
                    batched_solveconfigs,
                    batched_states,
                    batched_target_q,
                    batched_actions,
                    batched_path_actions,
                    batched_trajectory_indices,
                    batched_step_indices,
                    batched_weights,
                ),
            )
            return (q_params, opt_state), (losses, auxs)

        replay_keys = jax.random.split(key, replay_ratio)
        (q_params, opt_state,), (losses, auxs) = jax.lax.scan(
            replay_loop,
            (q_params, opt_state),
            replay_keys,
        )
        loss = jnp.mean(losses)
        return (
            q_params,
            opt_state,
            loss,
            auxs,
        )

    if n_devices > 1:

        def pmap_qfunction_train(key, dataset, q_params, target_q_params, opt_state):
            keys = jax.random.split(key, n_devices)
            (qfunc_params, opt_state, loss, auxs,) = jax.pmap(
                qfunction_train, in_axes=(0, 0, None, None, None), axis_name="devices"
            )(keys, dataset, q_params, target_q_params, opt_state)
            qfunc_params = jax.tree_util.tree_map(lambda xs: xs[0], qfunc_params)
            opt_state = jax.tree_util.tree_map(lambda xs: xs[0], opt_state)
            loss = jnp.mean(loss)
            return (
                qfunc_params,
                opt_state,
                loss,
                auxs,
            )

        return pmap_qfunction_train
    else:
        return jax.jit(qfunction_train)
