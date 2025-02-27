import math
from typing import Callable

import chex
import jax
import jax.numpy as jnp
import optax

from puzzle.world_model.world_model_puzzle_base import WorldModelPuzzleBase


def sigmoid_loss_fn(preds: chex.Array, labels: chex.Array) -> chex.Array:
    """
    Binary cross entropy loss function for sigmoid outputs.

    Args:
        preds: Predictions after sigmoid activation (values between 0 and 1)
        labels: Binary labels (0 or 1)

    Returns:
        Binary cross entropy loss
    """
    # Clip predictions to avoid numerical instability
    preds = jnp.clip(preds, 1e-7, 1 - 1e-7)
    return -jnp.mean(labels * jnp.log(preds) + (1 - labels) * jnp.log(1 - preds))


def accuracy_fn(preds: chex.Array, labels: chex.Array) -> chex.Array:
    return jnp.mean(jnp.sum(jnp.logical_xor(preds, labels), axis=tuple(range(1, preds.ndim))) == 0)


def world_model_train_builder(
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
            (latent, rounded_latent, decoded),
            (next_latent, rounded_next_latent, next_decoded),
            (forward_latent_pred, rounded_forward_latent_pred),
            (backward_latent_pred, rounded_backward_latent_pred),
        ), variable_updates = train_info_fn(params, data, next_data, action, training=True)
        new_params = {"params": params["params"], "batch_stats": variable_updates["batch_stats"]}
        data_scaled = (data / 255.0) * 2 - 1
        next_data_scaled = (next_data / 255.0) * 2 - 1
        AE_loss = jnp.mean(
            0.5 * optax.l2_loss(data_scaled, decoded)
            + 0.5 * optax.l2_loss(next_data_scaled, next_decoded)
        )

        forward_loss = jnp.mean(
            0.5 * sigmoid_loss_fn(next_latent, jax.lax.stop_gradient(rounded_forward_latent_pred))
            + 0.5 * sigmoid_loss_fn(forward_latent_pred, jax.lax.stop_gradient(rounded_next_latent))
        )

        backward_loss = jnp.mean(
            0.5 * sigmoid_loss_fn(latent, jax.lax.stop_gradient(rounded_backward_latent_pred))
            + 0.5 * sigmoid_loss_fn(backward_latent_pred, jax.lax.stop_gradient(rounded_latent))
        )

        total_loss = (1 - loss_weight) * AE_loss + loss_weight * (forward_loss + backward_loss)
        forward_accuracy = accuracy_fn(rounded_forward_latent_pred, rounded_next_latent)
        backward_accuracy = accuracy_fn(rounded_backward_latent_pred, rounded_latent)
        return total_loss, (
            new_params,
            AE_loss,
            forward_loss,
            backward_loss,
            forward_accuracy,
            backward_accuracy,
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
            (
                loss,
                (params, AE_loss, forward_loss, backward_loss, forward_accuracy, backward_accuracy),
            ), grads = jax.value_and_grad(loss_fn, has_aux=True)(
                params,
                states,
                next_states,
                actions,
                loss_weight,
            )
            updates, opt_state = optimizer.update(grads, opt_state, params=params)
            params = optax.apply_updates(params, updates)
            return (params, opt_state), (
                loss,
                AE_loss,
                forward_loss,
                backward_loss,
                forward_accuracy,
                backward_accuracy,
            )

        (params, opt_state), (
            losses,
            AE_losses,
            forward_losses,
            backward_losses,
            forward_accuracies,
            backward_accuracies,
        ) = jax.lax.scan(
            train_loop,
            (params, opt_state),
            (batched_states, batched_next_states, batched_actions),
        )
        loss = jnp.mean(losses)
        AE_loss = jnp.mean(AE_losses)
        forward_loss = jnp.mean(forward_losses)
        backward_loss = jnp.mean(backward_losses)
        forward_accuracy = jnp.mean(forward_accuracies)
        backward_accuracy = jnp.mean(backward_accuracies)
        return (
            params,
            opt_state,
            loss,
            AE_loss,
            forward_loss,
            backward_loss,
            forward_accuracy,
            backward_accuracy,
        )

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
                (_, rounded_latent, _),
                (_, rounded_next_latent, _),
                (_, rounded_forward_latent_pred),
                (_, rounded_backward_latent_pred),
            ), _ = train_info_fn(params, states, next_states, actions, training=False)
            forward_accuracy = accuracy_fn(rounded_forward_latent_pred, rounded_next_latent)
            backward_accuracy = accuracy_fn(rounded_backward_latent_pred, rounded_latent)
            return None, (forward_accuracy, backward_accuracy)

        _, (forward_accuracies, backward_accuracies) = jax.lax.scan(
            eval_loop,
            None,
            (batched_states, batched_next_states, batched_actions),
        )
        return jnp.mean(forward_accuracies), jnp.mean(backward_accuracies)

    return jax.jit(eval_fn)
