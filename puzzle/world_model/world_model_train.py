import math
from typing import Callable

import chex
import jax
import jax.numpy as jnp
import optax

from puzzle.world_model.world_model_puzzle_base import WorldModelPuzzleBase


def binary_cross_entropy(logits: chex.Array, labels: chex.Array) -> chex.Array:
    return -jnp.sum(
        labels * jax.nn.log_sigmoid(logits) + (1 - labels) * jax.nn.log_sigmoid(-logits),
        axis=tuple(range(1, logits.ndim)),
    )


def l2_loss(preds: chex.Array, labels: chex.Array) -> chex.Array:
    return jnp.mean(jnp.square(preds - labels), axis=tuple(range(1, preds.ndim)))


def world_model_train_builder(
    minibatch_size: int,
    train_info_fn: Callable,
    optimizer: optax.GradientTransformation = optax.adam(1e-4),
    loss_weight: float = 0.5,
):
    def loss_fn(
        params: jax.tree_util.PyTreeDef,
        data: chex.Array,
        next_data: chex.Array,
        action: chex.Array,
    ):
        (
            _,
            _,
            decoded,
            next_latent,
            rounded_next_latent,
            next_decoded,
            next_latent_preds,
            rounded_next_latent_preds,
        ), variable_updates = train_info_fn(params, data, next_data, training=True)
        params["batch_stats"] = variable_updates["batch_stats"]
        data_scaled = (data / 255.0) * 2 - 1
        next_data_scaled = (next_data / 255.0) * 2 - 1
        AE_loss = jnp.mean(
            jnp.square(data_scaled - decoded) + jnp.square(next_data_scaled - next_decoded)
        )
        action = jnp.reshape(
            action, (-1,) + (1,) * (next_latent_preds.ndim - 1)
        )  # [batch_size, 1, ...]
        next_latent_pred = jnp.take_along_axis(next_latent_preds, action, axis=1).squeeze(
            axis=1
        )  # [batch_size, ...]
        rounded_next_latent_pred = jnp.take_along_axis(
            rounded_next_latent_preds, action, axis=1
        ).squeeze(
            axis=1
        )  # [batch_size, ...]
        WM_loss = jnp.mean(
            l2_loss(next_latent, jax.lax.stop_gradient(rounded_next_latent_pred))
            + l2_loss(next_latent_pred, jax.lax.stop_gradient(rounded_next_latent))
        )
        return (1 - loss_weight) * AE_loss + loss_weight * WM_loss, (params, AE_loss, WM_loss)

    def train_fn(
        key: chex.PRNGKey,
        dataset: tuple[
            WorldModelPuzzleBase.State, WorldModelPuzzleBase.State, chex.Array
        ],  # (state, next_state, action)
        params: jax.tree_util.PyTreeDef,
        opt_state: optax.OptState,
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

        def train_loop(carry, batched_dataset):
            params, opt_state = carry
            states, next_states, actions = batched_dataset
            (loss, (params, AE_loss, WM_loss)), grads = jax.value_and_grad(loss_fn, has_aux=True)(
                params,
                states,
                next_states,
                actions,
            )
            updates, opt_state = optimizer.update(grads, opt_state, params=params)
            params = optax.apply_updates(params, updates)
            return (params, opt_state), (loss, AE_loss, WM_loss)

        (params, opt_state), (losses, AE_losses, WM_losses) = jax.lax.scan(
            train_loop,
            (params, opt_state),
            (batched_states, batched_next_states, batched_actions),
        )
        loss = jnp.mean(losses)
        AE_loss = jnp.mean(AE_losses)
        WM_loss = jnp.mean(WM_losses)
        return params, opt_state, loss, AE_loss, WM_loss

    return jax.jit(train_fn)
