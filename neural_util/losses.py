import jax
import jax.numpy as jnp


def hubberloss(x, delta):
    abs_errors = jnp.abs(x)
    quadratic = jnp.minimum(abs_errors, delta)
    # Same as max(abs_x - delta, 0) but avoids potentially doubling gradient.
    linear = abs_errors - quadratic
    return 0.5 * quadratic**2 + delta * linear


def quantile_weighted_huber_loss(
    current_value: jax.Array,
    target_value: jax.Array,
    delta: float = 1.0,
    quantile: float = 0.5,
):
    diff = target_value - current_value  # [batch_size]
    huber_loss = hubberloss(diff, delta)  # [batch_size]
    error_neg = (diff < 0.0).astype(jnp.float32)  # [batch_size]
    weight = jax.lax.stop_gradient(jnp.abs(quantile - error_neg))  # [batch_size]
    return jnp.mean(huber_loss * weight)  # scalar
