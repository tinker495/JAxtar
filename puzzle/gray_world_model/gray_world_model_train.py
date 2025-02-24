import math
from typing import Callable

import chex
import jax
import jax.numpy as jnp
import optax

from puzzle.world_model.world_model_puzzle_base import WorldModelPuzzleBase


def accuracy_fn(preds: chex.Array, labels: chex.Array) -> chex.Array:
    return jnp.mean(jnp.sum(jnp.logical_xor(preds, labels), axis=tuple(range(1, preds.ndim))) == 0)


def gray_world_model_train_builder(
    minibatch_size: int,
    train_info_fn: Callable,
    optimizer: optax.GradientTransformation = optax.adam(1e-4),
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
            rounded_latent,
            decoded,
            next_latent,
            rounded_next_latent,
            next_decoded,
            flipped,
        ), variable_updates = train_info_fn(params, data, next_data, training=True)
        params["batch_stats"] = variable_updates["batch_stats"]
        data_scaled = (data / 255.0) * 2 - 1
        next_data_scaled = (next_data / 255.0) * 2 - 1
        AE_loss = jnp.mean(
            0.5 * optax.l2_loss(data_scaled, decoded)
            + 0.5 * optax.l2_loss(next_data_scaled, next_decoded)
        )
        action = jnp.reshape(action, (-1,) + (1,) * (flipped.ndim - 1))  # [batch_size, 1, ...]
        next_flipped = jnp.take_along_axis(flipped, action, axis=1).squeeze(
            axis=1
        )  # [batch_size, latent_size + 1]
        flipped_one_hot = jax.nn.one_hot(jnp.argmax(flipped, axis=-1), next_flipped.shape[-1])[
            ..., :-1
        ]  # (batch_size, action_size, latent_size)
        pred_latent = jnp.logical_xor(next_flipped, flipped_one_hot)
        next_rounded_latent = jnp.take_along_axis(rounded_next_latent, action, axis=1).squeeze(
            axis=1
        )  # [batch_size, latent_size]
        pred_diff = jnp.logical_xor(rounded_latent, next_rounded_latent)
        WM_loss = jnp.mean(
            0.5 * optax.l2_loss(next_latent, jax.lax.stop_gradient(pred_latent))
            + 0.5 * optax.l2_loss(pred_diff, jax.lax.stop_gradient(pred_diff))
        )
        total_loss = (1 - loss_weight) * AE_loss + loss_weight * WM_loss
        accuracy = accuracy_fn(rounded_next_latent, pred_latent)
        return total_loss, (
            params,
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

        batch_indexs = jnp.concatenate(
            [
                jax.random.permutation(key, jnp.arange(data_size)),
                jax.random.randint(key, (batch_size * minibatch_size - data_size,), 0, data_size),
            ],
            axis=0,
        )  # [batch_size * minibatch_size]
        batch_indexs = jnp.reshape(batch_indexs, (batch_size, minibatch_size))

        batched_states = jnp.take(states, batch_indexs, axis=0)
        batched_next_states = jnp.take(next_states, batch_indexs, axis=0)
        batched_actions = jnp.take(actions, batch_indexs, axis=0)
        loss_weight = jnp.clip((epoch - 10) / 100.0, 0.0, 1.0) * 0.5

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


def gray_world_model_eval_builder(
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
                rounded_next_latent,
                _,
                next_latent_preds,
                rounded_next_latent_preds,
            ), _ = train_info_fn(params, states, next_states, training=False)

            actions = jnp.reshape(
                actions, (-1,) + (1,) * (next_latent_preds.ndim - 1)
            )  # [batch_size, 1, ...]
            rounded_next_latent_pred = jnp.take_along_axis(
                rounded_next_latent_preds, actions, axis=1
            ).squeeze(
                axis=1
            )  # [batch_size, ...]
            accuracy = accuracy_fn(rounded_next_latent, rounded_next_latent_pred)
            return None, accuracy

        _, accuracies = jax.lax.scan(
            eval_loop,
            None,
            (batched_states, batched_next_states, batched_actions),
        )
        return jnp.mean(accuracies)

    return jax.jit(eval_fn)
