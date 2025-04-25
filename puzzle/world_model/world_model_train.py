import math
from typing import Callable

import chex
import jax
import jax.numpy as jnp
import optax

from puzzle.world_model.world_model_puzzle_base import WorldModelPuzzleBase


def accuracy_fn(preds: chex.Array, labels: chex.Array) -> chex.Array:
    return jnp.mean(jnp.sum(jnp.logical_xor(preds, labels), axis=tuple(range(1, preds.ndim))) == 0)


def world_model_train_builder(
    minibatch_size: int,
    train_info_fn: Callable,
    optimizer: optax.GradientTransformation = optax.adam(1e-4),
    loss_ratio: float = 0.1,
):
    def loss_fn(
        params: jax.tree_util.PyTreeDef,
        data: chex.Array,
        next_data: chex.Array,
        action: chex.Array,
        loss_weight: float = 0.5,
    ):
        (
            _,
            _,
            decoded,
            next_logits,
            rounded_next_latents,
            next_decoded,
            next_logits_pred,
            rounded_next_latents_pred,
        ), variable_updates = train_info_fn(params, data, next_data, action, training=True)
        new_params = {"params": params["params"], "batch_stats": variable_updates["batch_stats"]}
        data_scaled = (data / 255.0) * 2 - 1
        next_data_scaled = (next_data / 255.0) * 2 - 1
        AE_loss = jnp.mean(
            0.5 * optax.l2_loss(data_scaled, decoded)
            + 0.5 * optax.l2_loss(next_data_scaled, next_decoded)
        )

        WM_loss = jnp.mean(
            0.5
            * optax.sigmoid_binary_cross_entropy(
                next_logits, jax.lax.stop_gradient(rounded_next_latents_pred)
            )
            + 0.5
            * optax.sigmoid_binary_cross_entropy(
                next_logits_pred, jax.lax.stop_gradient(rounded_next_latents)
            )
        )
        total_loss = (1 - loss_weight) * AE_loss + loss_weight * WM_loss
        accuracy = accuracy_fn(rounded_next_latents, rounded_next_latents_pred)
        return total_loss, (
            new_params,
            AE_loss,
            WM_loss,
            accuracy,
        )

    def train_fn(
        key: chex.PRNGKey,
        dataset: tuple[
            WorldModelPuzzleBase.State, WorldModelPuzzleBase.State, chex.Array
        ],  # (state, next_state, action)
        params: jax.tree_util.PyTreeDef,
        opt_state: optax.OptState,
        epoch: int,
    ):
        """
        Q-learning is a heuristic for the sliding puzzle problem.
        """
        states, next_states, actions = dataset
        data_size = actions.shape[0]
        batch_size = math.ceil(data_size / minibatch_size)

        key, subkey = jax.random.split(key)
        batch_indexs = jnp.concatenate(
            [
                jax.random.permutation(key, jnp.arange(data_size)),
                jax.random.randint(
                    subkey, (batch_size * minibatch_size - data_size,), 0, data_size
                ),
            ],
            axis=0,
        )  # [batch_size * minibatch_size]
        batch_indexs = jnp.reshape(batch_indexs, (batch_size, minibatch_size))

        batched_states = jnp.take(states, batch_indexs, axis=0)
        batched_next_states = jnp.take(next_states, batch_indexs, axis=0)
        batched_actions = jnp.take(actions, batch_indexs, axis=0)
        loss_weight = jnp.clip((epoch - 100) / 1000.0, 0.0001, 1.0) * loss_ratio

        def train_loop(carry, batched_dataset):
            params, opt_state = carry
            states, next_states, actions = batched_dataset
            (loss, (params, AE_loss, WM_loss, accuracy)), grads = jax.value_and_grad(
                loss_fn, has_aux=True
            )(
                params,
                states,
                next_states,
                actions,
                loss_weight,
            )
            updates, opt_state = optimizer.update(grads, opt_state, params=params)
            params = optax.apply_updates(params, updates)
            return (params, opt_state), (loss, AE_loss, WM_loss, accuracy)

        (params, opt_state), (losses, AE_losses, WM_losses, accuracies) = jax.lax.scan(
            train_loop,
            (params, opt_state),
            (batched_states, batched_next_states, batched_actions),
        )
        loss = jnp.mean(losses)
        AE_loss = jnp.mean(AE_losses)
        WM_loss = jnp.mean(WM_losses)
        accuracy = jnp.mean(accuracies)
        return params, opt_state, loss, AE_loss, WM_loss, accuracy

    return jax.jit(train_fn)


def world_model_eval_builder(
    train_info_fn: Callable,
    minibatch_size: int,
):
    def eval_fn(
        params: jax.tree_util.PyTreeDef,
        trajetory: tuple[chex.Array, chex.Array],
    ):
        states_all, actions = trajetory
        data_size = actions.shape[0]
        batch_size = math.ceil(data_size / minibatch_size)

        states = states_all[:-1]
        next_states = states_all[1:]

        batch_indexs = jnp.arange(data_size)
        batch_indexs = jnp.reshape(batch_indexs, (batch_size, minibatch_size))

        batched_states = jnp.take(states, batch_indexs, axis=0)
        batched_next_states = jnp.take(next_states, batch_indexs, axis=0)
        batched_actions = jnp.take(actions, batch_indexs, axis=0)

        def eval_loop(_, batched_dataset):
            states, next_states, actions = batched_dataset
            (
                _,
                _,
                _,
                _,
                rounded_next_latents,
                _,
                _,
                rounded_next_latents_pred,
            ), _ = train_info_fn(params, states, next_states, actions, training=False)
            accuracy = accuracy_fn(rounded_next_latents, rounded_next_latents_pred)
            return None, accuracy

        _, accuracies = jax.lax.scan(
            eval_loop,
            None,
            (batched_states, batched_next_states, batched_actions),
        )
        return jnp.mean(accuracies)

    return jax.jit(eval_fn)
