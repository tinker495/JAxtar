import math
from typing import Callable

import chex
import jax
import jax.numpy as jnp
import optax

from puzzle.world_model.world_model_puzzle_base import WorldModelPuzzleBase


def round_through_gradient(x: chex.Array) -> chex.Array:
    # x is a sigmoided value in the range [0, 1]. Use a straight-through estimator:
    # the forward pass returns jnp.round(x) while the gradient flows as if it were the identity.
    return x + jax.lax.stop_gradient(jnp.round(x) - x)


def world_model_train_builder(
    minibatch_size: int,
    auto_encoder_encode_fn: Callable,
    auto_encoder_decode_fn: Callable,
    world_model_transition_fn: Callable,
    optimizer: optax.GradientTransformation = optax.adam(1e-4),
    loss_weight: float = 0.5,
):
    def loss_fn(
        params: jax.tree_util.PyTreeDef,
        data: chex.Array,
        next_data: chex.Array,
        action: chex.Array,
    ):
        latent = auto_encoder_encode_fn(params, data, training=True)
        rounded_latent = round_through_gradient(latent)
        decoded = auto_encoder_decode_fn(params, rounded_latent, training=True)
        next_latent = auto_encoder_encode_fn(params, next_data, training=True)
        rounded_next_latent = round_through_gradient(next_latent)
        next_decoded = auto_encoder_decode_fn(params, rounded_next_latent, training=True)
        AE_loss = jnp.mean(jnp.square(data - decoded) + jnp.square(next_data - next_decoded))

        next_latents_pred = world_model_transition_fn(params, rounded_latent, training=True)
        next_latents_pred = round_through_gradient(next_latents_pred)
        action = jnp.reshape(
            action, (-1,) + (1,) * (next_latents_pred.ndim - 1)
        )  # [batch_size, 1, ...]
        next_latent_pred = jnp.take_along_axis(next_latents_pred, action, axis=1).squeeze(
            axis=1
        )  # [batch_size, ...]
        WM_loss = jnp.mean(
            jnp.square(next_latent_pred - jax.lax.stop_gradient(rounded_next_latent))
            + jnp.square(jax.lax.stop_gradient(next_latent_pred) - rounded_next_latent)
        )

        return (1 - loss_weight) * AE_loss + loss_weight * WM_loss

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
            loss, grads = jax.value_and_grad(loss_fn, has_aux=True)(
                params,
                states,
                next_states,
                actions,
            )
            updates, opt_state = optimizer.update(grads, opt_state, params=params)
            params = optax.apply_updates(params, updates)
            return (params, opt_state), loss

        (params, opt_state), losses = jax.lax.scan(
            train_loop,
            (params, opt_state),
            (batched_states, batched_next_states, batched_actions),
        )
        loss = jnp.mean(losses)
        return params, opt_state, loss

    return jax.jit(train_fn)
