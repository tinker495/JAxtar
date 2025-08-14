from typing import Literal

import jax
import jax.numpy as jnp


def mse_loss(diff: jnp.ndarray) -> jnp.ndarray:
    return jnp.square(diff)


def huber_loss(diff: jnp.ndarray, delta: float = 0.1) -> jnp.ndarray:
    abs_diff = jnp.abs(diff)
    quadratic = 0.5 * jnp.square(diff)
    linear = delta * (abs_diff - 0.5 * delta)
    return jnp.where(abs_diff <= delta, quadratic, linear)


def logcosh_loss(diff: jnp.ndarray) -> jnp.ndarray:
    # Stable formulation: log(cosh(x)) = x + softplus(-2x) - log(2)
    return diff + jax.nn.softplus(-2.0 * diff) - jnp.log(2.0)


def loss_from_diff(
    diff: jnp.ndarray,
    loss: Literal["mse", "huber", "logcosh"] = "mse",
    huber_delta: float = 0.1,
) -> jnp.ndarray:
    if loss == "mse":
        return mse_loss(diff)
    if loss == "huber":
        return huber_loss(diff, delta=huber_delta)
    if loss == "logcosh":
        return logcosh_loss(diff)
    # Fallback to MSE if invalid option sneaks in
    return mse_loss(diff)
