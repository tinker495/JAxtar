import math
from typing import Any, Callable, Optional

import chex
import jax
import jax.numpy as jnp
import optax
import xtructure.numpy as xnp

from neural_util.basemodel import DistanceHLGModel, DistanceModel
from train_util.util import (
    apply_with_conditional_batch_stats,
    build_new_params_from_updates,
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
        solveconfigs: chex.Array,
        states: chex.Array,
        actions: chex.Array,
        target_qs: chex.Array,
        weights: chex.Array,
        key: chex.PRNGKey,
    ):
        # Preprocess during training
        preproc = jax.vmap(preproc_fn)(solveconfigs, states)
        per_sample_loss, variable_updates = apply_with_conditional_batch_stats(
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
            rngs={"params": key},
        )
        new_params = build_new_params_from_updates(q_params, variable_updates)
        loss_value = jnp.mean(per_sample_loss.squeeze() * weights)
        return loss_value, new_params

    def qfunction_train(
        key: chex.PRNGKey,
        dataset: dict[str, chex.Array],
        q_params: Any,
        opt_state: optax.OptState,
    ):
        """Run one optimization epoch of neural Q-learning for the provided puzzle dataset."""
        solveconfigs = dataset["solveconfigs"]
        states = dataset["states"]
        target_q = dataset["target_q"]
        actions = dataset["actions"]
        data_size = target_q.shape[0]
        batch_size = math.ceil(data_size / minibatch_size)

        loss_weights = jnp.ones(data_size)
        loss_weights = loss_weights / jnp.mean(loss_weights)

        def train_loop(carry, batched_dataset):
            q_params, opt_state, key = carry
            step_key, key = jax.random.split(key)
            solveconfigs, states, target_q, actions, weights = batched_dataset
            (loss, q_params), grads = jax.value_and_grad(qfunction_train_loss, has_aux=True)(
                q_params,
                solveconfigs,
                states,
                actions,
                target_q,
                weights,
                step_key,
            )
            if n_devices > 1:
                grads = jax.lax.psum(grads, axis_name="devices")
            updates, opt_state = optimizer.update(grads, opt_state, params=q_params)
            q_params = optax.apply_updates(q_params, updates)
            return (q_params, opt_state, key), loss

        def replay_loop(carry, replay_key):
            q_params, opt_state = carry

            key_perm, key_fill, key_train = jax.random.split(replay_key, 3)
            batch_indexs = jnp.concatenate(
                [
                    jax.random.permutation(key_perm, jnp.arange(data_size)),
                    jax.random.randint(
                        key_fill,
                        (batch_size * minibatch_size - data_size,),
                        0,
                        data_size,
                    ),
                ],
                axis=0,
            )
            batch_indexs = jnp.reshape(batch_indexs, (batch_size, minibatch_size))

            batched_solveconfigs = xnp.take(solveconfigs, batch_indexs, axis=0)
            batched_states = xnp.take(states, batch_indexs, axis=0)
            batched_target_q = jnp.take(target_q, batch_indexs, axis=0)
            batched_actions = jnp.take(actions, batch_indexs, axis=0)
            batched_weights = jnp.take(loss_weights, batch_indexs, axis=0)
            batched_weights = batched_weights / (
                jnp.mean(batched_weights, axis=1, keepdims=True) + 1e-8
            )

            (q_params, opt_state, _), losses = jax.lax.scan(
                train_loop,
                (q_params, opt_state, key_train),
                (
                    batched_solveconfigs,
                    batched_states,
                    batched_target_q,
                    batched_actions,
                    batched_weights,
                ),
            )
            return (q_params, opt_state), losses

        replay_keys = jax.random.split(key, replay_ratio)
        (q_params, opt_state,), losses = jax.lax.scan(
            replay_loop,
            (q_params, opt_state),
            replay_keys,
        )
        loss = jnp.mean(losses)
        return (
            q_params,
            opt_state,
            loss,
        )

    if n_devices > 1:

        def pmap_qfunction_train(key, dataset, q_params, opt_state):
            keys = jax.random.split(key, n_devices)
            (qfunc_params, opt_state, loss,) = jax.pmap(
                qfunction_train, in_axes=(0, 0, None, None), axis_name="devices"
            )(keys, dataset, q_params, opt_state)
            qfunc_params = jax.tree_util.tree_map(lambda xs: xs[0], qfunc_params)
            opt_state = jax.tree_util.tree_map(lambda xs: xs[0], opt_state)
            loss = jnp.mean(loss)
            return (
                qfunc_params,
                opt_state,
                loss,
            )

        return pmap_qfunction_train
    else:
        return jax.jit(qfunction_train)
