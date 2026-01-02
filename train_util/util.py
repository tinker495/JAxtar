from typing import Any, Callable

import chex
import jax
import jax.numpy as jnp
from flax import linen as nn


def get_self_predictive_train_args(
    model: nn.Module,
    ema_target_heuristic_params: Any,
    preprocessed_states: chex.Array,  # (batch_size, path_length, state_dim)
    trajectory_indices: chex.Array,  # (batch_size, path_length)
    step_indices: chex.Array,  # (batch_size, path_length)
) -> tuple[Any, Any]:

    if hasattr(model, "states_to_latents"):
        next_preprocessed_states = preprocessed_states[
            :, :-1
        ]  # (batch_size, path_length - 1, state_dim)
        ema_next_state_latents = model.apply(
            ema_target_heuristic_params,
            next_preprocessed_states,
            training=True,
            method=model.states_to_latents,
        )  # (batch_size, path_length - 1, latent_dim)
        ema_next_state_projection = model.apply(
            ema_target_heuristic_params,
            ema_next_state_latents,
            training=True,
            method=model.latents_to_projection,
        )  # (batch_size, path_length - 1, projection_dim)
        ema_next_state_projection = jnp.float32(ema_next_state_projection)
        same_trajectory_masks = (
            trajectory_indices[:, :-1] == trajectory_indices[:, -1][:, jnp.newaxis]
        )  # (batch_size, path_length - 1)
        return ema_next_state_projection, same_trajectory_masks
    else:
        return None, None


def round_through_gradient(x: chex.Array) -> chex.Array:
    # x is a sigmoided value in the range [0, 1]. Use a straight-through estimator:
    # the forward pass returns jnp.round(x) while the gradient flows as if it were the identity.
    return x + jax.lax.stop_gradient(jnp.where(x > 0.5, 1.0, 0.0).astype(jnp.float32) - x)


def boltzmann_action_selection(
    q_values: chex.Array,
    temperature: float = 1.0 / 3.0,
    epsilon: float = 0.1,
) -> chex.Array:
    # Determine valid entries before sanitizing infinities
    mask = jnp.isfinite(q_values)
    q_values = jnp.nan_to_num(q_values, posinf=1e6, neginf=-1e6)

    # Scale Q-values by temperature for softmax
    safe_temperature = jnp.maximum(temperature, 1e-8)
    scaled_q_values = -q_values / safe_temperature

    # Apply mask before softmax to avoid overflow
    masked_q_values = jnp.where(mask, scaled_q_values, -jnp.inf)
    probs = jax.nn.softmax(masked_q_values, axis=1)
    probs = jnp.where(mask, probs, 0.0)

    # Row-wise normalization with guard
    row_sum = jnp.sum(probs, axis=1, keepdims=True)
    probs = jnp.where(row_sum > 0.0, probs / row_sum, probs)

    # Calculate uniform probabilities
    valid_actions = jnp.sum(mask, axis=1, keepdims=True)
    uniform_valid = jnp.where(mask, 1.0 / jnp.maximum(valid_actions, 1.0), 0.0)

    action_size = q_values.shape[1]
    uniform_all = jnp.ones_like(probs) / jnp.maximum(action_size, 1)

    # Fallback if no valid actions in a row
    probs = jnp.where(valid_actions > 0, probs, uniform_all)

    # ε-greedy mixing and final guard renormalization
    probs = probs * (1.0 - epsilon) + uniform_valid * epsilon
    probs = probs / (jnp.sum(probs, axis=1, keepdims=True) + 1e-8)
    return probs


def apply_with_conditional_batch_stats(
    apply_fn: Callable,
    params: Any,
    *apply_args,
    training: bool,
    n_devices: int = 1,
    collection: str = "batch_stats",
    axis_name: str = "devices",
    **apply_kwargs,
):
    """
    Call a Flax Module.apply while conditionally enabling mutable batch_stats during training.

    Returns (outputs, variable_updates_dict), where variable_updates_dict is empty when
    no mutable collection is used.
    """
    if training and collection in params:
        outputs, variable_updates = apply_fn(
            params, *apply_args, training=True, mutable=[collection], **apply_kwargs
        )
        if n_devices > 1:
            variable_updates = jax.lax.pmean(variable_updates, axis_name=axis_name)
        return outputs, variable_updates
    else:
        outputs = apply_fn(params, *apply_args, training=training, **apply_kwargs)
        return outputs, {}


# Backward-compat alias (to be removed later)
apply_with_optional_batch_stats = apply_with_conditional_batch_stats


def build_new_params_from_updates(
    params: Any, variable_updates: dict, collection: str = "batch_stats"
):
    new_params = {"params": params["params"]}
    if collection in variable_updates:
        new_params[collection] = variable_updates[collection]
    return new_params
