from typing import Any, Literal, Mapping, Optional

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


def _quantile_weight(diff: jnp.ndarray, tau: float) -> jnp.ndarray:
    tau = jnp.clip(tau, 1e-6, 1.0 - 1e-6)
    indicator = (diff < 0).astype(diff.dtype)
    return jnp.abs(tau - indicator)


def asymmetric_huber_loss(
    diff: jnp.ndarray,
    delta: float = 0.1,
    tau: float = 0.25,
) -> jnp.ndarray:
    base_loss = huber_loss(diff, delta=delta)
    weights = _quantile_weight(diff, tau=tau) * 2.0 # double the weight of the loss
    return weights * base_loss


def asymmetric_logcosh_loss(diff: jnp.ndarray, tau: float = 0.25) -> jnp.ndarray:
    base_loss = logcosh_loss(diff)
    weights = _quantile_weight(diff, tau=tau)
    return weights * base_loss


def loss_from_diff(
    diff: jnp.ndarray,
    loss: Literal[
        "mse",
        "huber",
        "logcosh",
        "asymmetric_huber",
        "asymmetric_logcosh",
    ] = "mse",
    loss_args: Optional[Mapping[str, Any]] = None,
) -> jnp.ndarray:
    args = loss_args or {}
    huber_delta = float(args.get("huber_delta", 0.1))
    asymmetric_tau = float(args.get("asymmetric_tau", 0.25))
    if loss == "mse":
        return mse_loss(diff)
    if loss == "huber":
        return huber_loss(diff, delta=huber_delta)
    if loss == "logcosh":
        return logcosh_loss(diff)
    if loss == "asymmetric_huber":
        return asymmetric_huber_loss(diff, delta=huber_delta, tau=asymmetric_tau)
    if loss == "asymmetric_logcosh":
        return asymmetric_logcosh_loss(diff, tau=asymmetric_tau)
    # Fallback to MSE if invalid option sneaks in
    return mse_loss(diff)
