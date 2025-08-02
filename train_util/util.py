import chex
import jax
import jax.numpy as jnp


def round_through_gradient(x: chex.Array) -> chex.Array:
    # x is a sigmoided value in the range [0, 1]. Use a straight-through estimator:
    # the forward pass returns jnp.round(x) while the gradient flows as if it were the identity.
    return x + jax.lax.stop_gradient(jnp.where(x > 0.5, 1.0, 0.0).astype(jnp.float32) - x)
