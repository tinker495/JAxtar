import math
from typing import Callable

import chex
import jax
import jax.numpy as jnp
import optax

from puzzle.world_model.world_model_puzzle_base import WorldModelPuzzleBase


def accuracy_fn(preds: chex.Array, labels: chex.Array) -> chex.Array:
    return jnp.mean(jnp.sum(jnp.logical_xor(preds, labels), axis=tuple(range(1, preds.ndim))) == 0)


def similarity_loss_fn(A: chex.Array, B: chex.Array):
    """
    Compute similarity loss between two arrays.

    Args:
        A: First array [batch_size, n_features]
        B: Second array [batch_size, n_features]
    """
    B = jax.lax.stop_gradient(B)
    # Compute cosine similarity
    A_norm = jnp.sqrt(
        jnp.sum(A**2, axis=-1, keepdims=True) + 1e-8
    )  # Add epsilon for numerical stability
    B_norm = jnp.sqrt(
        jnp.sum(B**2, axis=-1, keepdims=True) + 1e-8
    )  # Add epsilon for numerical stability
    dot_product = jnp.sum(A * B, axis=-1, keepdims=True)
    similarity = dot_product / (A_norm * B_norm)
    similarity = similarity.squeeze(-1)  # Remove the last dimension after calculation
    # Convert to loss (1 - similarity)
    loss = 1.0 - similarity
    return jnp.mean(loss)


def orthonormality_loss_fn(latent: chex.Array, other_latent: chex.Array):
    """
    Compute orthonormality regularization loss between two arrays.

    Args:
        latent: First array [batch_size, n_features] corresponding to B_ω(s_i, a_i)
        other_latent: Second array [batch_size, n_features] corresponding to B_ω(s'_j, a'_j)
    """
    batch_size = latent.shape[0]

    # Calculate dot products between all pairs
    dot_products = jnp.einsum("bd,cd->bc", latent, other_latent)  # [batch_size, batch_size]

    # First part of the regularization loss
    stopped_dot_products = jax.lax.stop_gradient(dot_products)
    stopped_other_latent = jax.lax.stop_gradient(other_latent)

    first_term = jnp.einsum(
        "bd,cd,bc->b", latent, stopped_other_latent, stopped_dot_products
    )  # [batch_size]
    first_term = jnp.sum(first_term) / (batch_size**2)

    # Second part: (1/b) * ∑_{i∈I} B_ω(s_i, a_i)^T * stop-gradient(B_ω(s_i, a_i))
    stopped_latent = jax.lax.stop_gradient(latent)
    second_term = jnp.sum(jnp.einsum("bd,bd->b", latent, stopped_latent)) / batch_size

    # Final orthonormality regularization loss
    return first_term - second_term


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
            (logits, rounded_latent, decoded),
            (next_logits, rounded_next_latent, next_decoded),
            (forward_logits_preds, rounded_forward_latent_pred),
            (
                projected_latent,
                next_projected_latent,
                forward_predicted_latents,
                backward_predicted_latents,
            ),
        ), variable_updates = train_info_fn(params, data, next_data, action, training=True)
        new_params = {"params": params["params"], "batch_stats": variable_updates["batch_stats"]}
        data_scaled = (data / 255.0) * 2 - 1
        next_data_scaled = (next_data / 255.0) * 2 - 1
        AE_loss = jnp.mean(
            0.5 * optax.l2_loss(data_scaled, decoded)
            + 0.5 * optax.l2_loss(next_data_scaled, next_decoded)
        )

        world_model_loss = jnp.mean(
            0.5
            * optax.sigmoid_binary_cross_entropy(
                next_logits, jax.lax.stop_gradient(rounded_forward_latent_pred)
            )
            + 0.5
            * optax.sigmoid_binary_cross_entropy(
                forward_logits_preds, jax.lax.stop_gradient(rounded_next_latent)
            )
        )

        forward_similarity = similarity_loss_fn(
            forward_predicted_latents,
            next_projected_latent,
        )

        similarity_loss = forward_similarity

        rolled_projected_latent = jnp.roll(projected_latent, 1, axis=0)

        # orthonormality regularization
        orthonormality_regularization_loss = (
            orthonormality_loss_fn(projected_latent, rolled_projected_latent) * 0.1
        )

        total_loss = (1 - loss_weight) * AE_loss + loss_weight * (
            world_model_loss + 0.01 * similarity_loss + 0.01 * orthonormality_regularization_loss
        )
        accuracy = accuracy_fn(rounded_forward_latent_pred, rounded_next_latent)
        return total_loss, (
            new_params,
            AE_loss,
            world_model_loss,
            similarity_loss,
            orthonormality_regularization_loss,
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
            (
                loss,
                (
                    params,
                    AE_loss,
                    world_model_loss,
                    similarity_loss,
                    orthonormality_regularization_loss,
                    accuracy,
                ),
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
                world_model_loss,
                similarity_loss,
                orthonormality_regularization_loss,
                accuracy,
            )

        (params, opt_state), (
            losses,
            AE_losses,
            world_model_losses,
            similarity_losses,
            orthonormality_regularization_losses,
            accuracies,
        ) = jax.lax.scan(
            train_loop,
            (params, opt_state),
            (batched_states, batched_next_states, batched_actions),
        )
        loss = jnp.mean(losses)
        AE_loss = jnp.mean(AE_losses)
        world_model_loss = jnp.mean(world_model_losses)
        similarity_loss = jnp.mean(similarity_losses)
        orthonormality_regularization_loss = jnp.mean(orthonormality_regularization_losses)
        accuracy = jnp.mean(accuracies)
        return (
            params,
            opt_state,
            loss,
            AE_loss,
            world_model_loss,
            similarity_loss,
            orthonormality_regularization_loss,
            accuracy,
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
                (_, _, _),
                (_, rounded_next_latent, _),
                (_, rounded_forward_latent_pred),
                (
                    projected_latent,
                    next_projected_latent,
                    forward_predicted_latents,
                    backward_predicted_latents,
                ),
            ), _ = train_info_fn(params, states, next_states, actions, training=False)
            accuracy = accuracy_fn(rounded_forward_latent_pred, rounded_next_latent)
            return None, (accuracy, projected_latent)

        _, (accuracies, projected_latents) = jax.lax.scan(
            eval_loop,
            None,
            (batched_states, batched_next_states, batched_actions),
        )
        projected_latents = projected_latents.reshape(-1, projected_latents.shape[-1])

        projected_latents = projected_latents[:1000]
        return jnp.mean(accuracies), projected_latents

    return jax.jit(eval_fn)
