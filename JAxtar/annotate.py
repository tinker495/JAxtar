import jax.numpy as jnp

KEY_DTYPE = jnp.float16
ACTION_DTYPE = jnp.uint16
HASH_SIZE_MULTIPLIER = 2  # Multiplier for hash table size to reduce collision probability
CUCKOO_TABLE_N = 2
