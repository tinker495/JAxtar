from typing import Any, Callable

import chex
import jax
import jax.numpy as jnp


def round_through_gradient(x: chex.Array) -> chex.Array:
    # x is a sigmoided value in the range [0, 1]. Use a straight-through estimator:
    # the forward pass returns jnp.round(x) while the gradient flows as if it were the identity.
    return x + jax.lax.stop_gradient(jnp.where(x > 0.5, 1.0, 0.0).astype(jnp.float32) - x)


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
