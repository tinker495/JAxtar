import math
from typing import Callable

import chex
import jax
import jax.numpy as jnp
import optax

from puzzle.world_model.world_model_puzzle_base import WorldModelPuzzleBase


def accuracy_fn(preds: chex.Array, labels: chex.Array) -> chex.Array:
    return jnp.mean(jnp.sum(jnp.logical_xor(preds, labels), axis=tuple(range(1, preds.ndim))) == 0)


def Q_fn(forward_projected_latent: chex.Array, backward_projected_target_latent: chex.Array):
    dot_products = jnp.einsum(
        "bd,cd->bc", forward_projected_latent, backward_projected_target_latent
    )  # [batch_size, batch_size]
    # dist = - jax.nn.log_sigmoid(dot_products)
    return dot_products  # [batch_size, batch_size]


def projection_distance_loss_fn(
    forward_projected_latent: chex.Array,
    backward_projected_target_latent: chex.Array,
    forward_projected_next_prim_latent_preds: chex.Array,
    backward_projected_target_prim_latent: chex.Array,
):
    # forward_projected_latent: [batch_size, latent_size]
    # forward_projected_next_prim_latent_preds: [batch_size, action_size, latent_size]
    # backward_projected_target_latent: [batch_size, latent_size]

    target_Qs = jax.vmap(Q_fn, in_axes=(0, None), out_axes=0)(
        forward_projected_next_prim_latent_preds, backward_projected_target_prim_latent
    )  # [action_size, batch_size, batch_size]
    target_Qs = jax.lax.stop_gradient(1.0 + jnp.min(target_Qs, axis=0))  # [batch_size, batch_size]
    target_Qs = jnp.maximum(target_Qs, 0.0)
    # Set diagonal elements (identity matrix positions) to zero
    batch_size = target_Qs.shape[0]
    target_Qs = target_Qs.at[jnp.diag_indices(batch_size)].set(0.0)
    # jax.debug.print("target_Qs: {output}", output=target_Qs[:10, :10])
    # this Qs is calculate heuristic distance between
    # forward_projected_latent and forward_projected_next_prim_latent_preds

    current_Qs = Q_fn(
        forward_projected_latent, backward_projected_target_latent
    )  # [batch_size, batch_size]
    loss = jnp.mean(optax.l2_loss(current_Qs, target_Qs))
    return loss, target_Qs, current_Qs


def self_projection_distance_loss_fn(
    forward_projected_latent: chex.Array, backward_projected_latent: chex.Array
):
    # forward_projected_latent: [batch_size, latent_size]
    # backward_projected_latent: [batch_size, latent_size]
    self_dot_products = jnp.sum(
        forward_projected_latent * backward_projected_latent, axis=-1
    )  # [batch_size]
    loss = jnp.mean(jnp.square(self_dot_products))
    return loss


def orthonormality_loss_fn(latent: chex.Array, other_latent: chex.Array):
    """
    Compute orthonormality regularization loss between two arrays.

    Args:
        latent: First array [batch_size, n_features] corresponding to P(s_i)
        other_latent: Second array [batch_size, n_features] corresponding to P(s'_j)
    """
    batch_size = latent.shape[0]

    # Calculate dot products between all pairs
    dot_products = jnp.einsum("bd,cd->bc", latent, other_latent)  # [batch_size, batch_size]

    # First part of the regularization loss - use jax functional style
    stopped_other_latent = jax.lax.stop_gradient(other_latent)
    stopped_dot_products = jax.lax.stop_gradient(dot_products)

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
    get_projections_fn: Callable,
    optimizer: optax.GradientTransformation = optax.adam(1e-4),
):
    def loss_fn(
        params: jax.tree_util.PyTreeDef,
        target_params: WorldModelPuzzleBase.State,
        data: chex.Array,
        next_data: chex.Array,
        action: chex.Array,
        loss_weight: float = 0.5,
    ):
        (
            (logits, rounded_latent, decoded),  # for AutoEncoder and WorldModel
            (next_logits, rounded_next_latent, next_decoded),  # for AutoEncoder and WorldModel
            (
                next_logits_pred,
                rounded_next_latent_pred,
                rounded_next_latent_preds,
            ),  # for WorldModel
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
                next_logits, jax.lax.stop_gradient(rounded_next_latent_pred)
            )
            + 0.5
            * optax.sigmoid_binary_cross_entropy(
                next_logits_pred, jax.lax.stop_gradient(rounded_next_latent)
            )
        )

        forward_projected_latent, backward_projected_latent = get_projections_fn(
            params, rounded_latent
        )
        _, backward_projected_target_latent = get_projections_fn(target_params, rounded_latent)
        forward_projected_next_latent_preds, _ = jax.vmap(
            get_projections_fn, in_axes=(None, 1), out_axes=0
        )(target_params, rounded_next_latent_preds)

        projection_distance_loss, target_Qs, current_Qs = projection_distance_loss_fn(
            forward_projected_latent,
            backward_projected_latent,
            forward_projected_next_latent_preds,
            backward_projected_target_latent,
        )

        total_projection_distance_loss = projection_distance_loss  # self_projection_distance_loss

        rolled_backward_projected_latent = jnp.roll(
            backward_projected_latent, minibatch_size // 2, axis=0
        )  # [batch_size, latent_size]
        # orthonormality regularization
        orthonormality_regularization_loss = orthonormality_loss_fn(
            backward_projected_latent, rolled_backward_projected_latent
        )

        total_loss = (1 - loss_weight) * AE_loss + loss_weight * (
            world_model_loss
            + total_projection_distance_loss
            + 0.01 * orthonormality_regularization_loss
        )
        accuracy = accuracy_fn(rounded_next_latent_pred, rounded_next_latent)

        return total_loss, (
            new_params,
            AE_loss,
            world_model_loss,
            total_projection_distance_loss,
            orthonormality_regularization_loss,
            accuracy,
            target_Qs,
            current_Qs,
        )

    def train_fn(
        key: chex.PRNGKey,
        dataset: tuple[
            WorldModelPuzzleBase.State, WorldModelPuzzleBase.State, chex.Array
        ],  # (state, next_state, action)
        params: jax.tree_util.PyTreeDef,
        target_params: WorldModelPuzzleBase.State,
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
                jnp.arange(data_size),
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

            # Use value_and_grad with has_aux=True to properly handle the auxiliary outputs
            (
                loss,
                (
                    params,
                    AE_loss,
                    world_model_loss,
                    total_projection_distance_loss,
                    orthonormality_regularization_loss,
                    accuracy,
                    target_Qs,
                    current_Qs,
                ),
            ), grads = jax.value_and_grad(loss_fn, has_aux=True)(
                params,
                target_params,
                states,
                next_states,
                actions,
                loss_weight,
            )

            # Ensure we're not capturing any tracers in the update step
            updates, new_opt_state = optimizer.update(grads, opt_state, params=params)
            new_params = optax.apply_updates(params, updates)

            return (new_params, new_opt_state), (
                loss,
                AE_loss,
                world_model_loss,
                total_projection_distance_loss,
                orthonormality_regularization_loss,
                accuracy,
                target_Qs,
                current_Qs,
            )

        (params, opt_state), (
            losses,
            AE_losses,
            world_model_losses,
            total_projection_distance_losses,
            orthonormality_regularization_losses,
            accuracies,
            target_Qs,
            current_Qs,
        ) = jax.lax.scan(
            train_loop,
            (params, opt_state),
            (batched_states, batched_next_states, batched_actions),
        )
        loss = jnp.mean(losses)
        AE_loss = jnp.mean(AE_losses)
        world_model_loss = jnp.mean(world_model_losses)
        total_projection_distance_loss = jnp.mean(total_projection_distance_losses)
        orthonormality_regularization_loss = jnp.mean(orthonormality_regularization_losses)
        accuracy = jnp.mean(accuracies)
        target_Qs = jnp.reshape(target_Qs, (-1,))
        current_Qs = jnp.reshape(current_Qs, (-1,))
        target_params = (
            params  # jax.tree_map(lambda x, y: 0.9 * x + 0.1 * y, target_params, params)
        )
        return (
            params,
            target_params,
            opt_state,
            loss,
            AE_loss,
            world_model_loss,
            total_projection_distance_loss,
            orthonormality_regularization_loss,
            accuracy,
            target_Qs,
            current_Qs,
        )

    return jax.jit(train_fn)


def world_model_eval_builder(
    train_info_fn: Callable,
    get_projections_fn: Callable,
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
                (logits, rounded_latent, decoded),  # for AutoEncoder and WorldModel
                (next_logits, rounded_next_latent, next_decoded),  # for AutoEncoder and WorldModel
                (
                    next_logits_pred,
                    rounded_next_latent_pred,
                    rounded_next_latent_preds,
                ),  # for WorldModel
            ), variable_updates = train_info_fn(
                params, states, next_states, actions, training=False
            )
            forward_projected_latent, backward_projected_latent = get_projections_fn(
                params, rounded_latent
            )
            accuracy = accuracy_fn(rounded_next_latent_pred, rounded_next_latent)
            return None, (accuracy, forward_projected_latent, backward_projected_latent)

        _, (accuracies, forward_projected_latents, backward_projected_latents) = jax.lax.scan(
            eval_loop,
            None,
            (batched_states, batched_next_states, batched_actions),
        )
        forward_projected_latents = forward_projected_latents.reshape(
            -1, forward_projected_latents.shape[-1]
        )
        backward_projected_latents = backward_projected_latents.reshape(
            -1, backward_projected_latents.shape[-1]
        )

        forward_projected_latents = forward_projected_latents[:1000]
        backward_projected_latents = backward_projected_latents[:1000]
        return jnp.mean(accuracies), forward_projected_latents, backward_projected_latents

    return jax.jit(eval_fn)
