import jax
import jax.numpy as jnp

KEY_DTYPE = jnp.bfloat16 if jax.default_backend() == "tpu" else jnp.float16
ACTION_DTYPE = jnp.uint8
HASH_SIZE_MULTIPLIER = 2  # Multiplier for hash table size to reduce collision probability
CUCKOO_TABLE_N = 2
