from typing import Any, Callable

import chex
import jax
import jax.numpy as jnp


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


def build_distance_train_loss(
    model: Any,
    preproc_fn: Callable,
    loss_type: str,
    loss_args: Any,
    n_devices: int = 1,
) -> Callable:
    """Build the per-minibatch train loss shared by distance-style trainers.

    The returned function has signature
    ``(params, batch_stats, solve_configs, states, *loss_inputs, key)`` and
    routes through ``model.train_loss`` with conditionally mutable batch_stats,
    returning ``(mean_loss, (new_batch_stats, log_infos))``.
    """

    def train_loss(
        params: Any,
        batch_stats: Any,
        solve_configs: chex.Array,
        states: chex.Array,
        *loss_inputs_and_key: Any,
    ):
        *loss_inputs, key = loss_inputs_and_key
        full_params = {"params": params}
        if batch_stats is not None:
            full_params["batch_stats"] = batch_stats

        preproc = jax.vmap(preproc_fn)(solve_configs, states)
        (per_sample_loss, log_infos), variable_updates = apply_with_conditional_batch_stats(
            model.apply,
            full_params,
            preproc,
            *loss_inputs,
            training=True,
            n_devices=n_devices,
            method=model.train_loss,
            loss_type=loss_type,
            loss_args=loss_args,
            rngs={"params": key},
        )
        new_batch_stats = variable_updates.get("batch_stats", batch_stats)
        return jnp.mean(per_sample_loss), (new_batch_stats, log_infos)

    return train_loss


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


def build_new_params_from_updates(
    params: Any, variable_updates: dict, collection: str = "batch_stats"
):
    new_params = {"params": params["params"]}
    if collection in variable_updates:
        new_params[collection] = variable_updates[collection]
    return new_params
