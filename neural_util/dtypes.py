import jax.numpy as jnp

DTYPE = jnp.bfloat16
# Use float32 for numerically sensitive heads / losses.
HEAD_DTYPE = jnp.float32
# Default parameter dtype (aligned to DTYPE for efficiency)
PARAM_DTYPE = jnp.float32
# Parameter dtype used when loading a checkpoint for inference (search / eval /
# benchmark via NeuralDistanceBase.load_model). The body matmuls already compute in
# DTYPE (bfloat16) and Flax downcasts f32 kernels to it at every apply, so storing the
# loaded params in bf16 does not change the body math but halves the parameter HBM
# bandwidth those matmuls are bound by. Set to None to keep PARAM_DTYPE (f32) on load.
INFERENCE_PARAM_DTYPE = jnp.bfloat16
