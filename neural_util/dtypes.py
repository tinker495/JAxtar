import jax.numpy as jnp

DTYPE = jnp.bfloat16
# Use float32 for numerically sensitive heads / losses.
HEAD_DTYPE = jnp.float32
# Default parameter dtype (aligned to DTYPE for efficiency)
PARAM_DTYPE = jnp.bfloat16
