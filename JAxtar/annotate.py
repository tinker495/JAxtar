import jax.numpy as jnp

KEY_DTYPE = jnp.float16
ACTION_DTYPE = jnp.uint8
HASH_POINT_DTYPE = jnp.uint32
HASH_TABLE_IDX_DTYPE = jnp.uint8
SIZE_DTYPE = jnp.uint32
HASH_SIZE_MULTIPLIER = 2  # Multiplier for hash table size to reduce collision probability
CUCKOO_TABLE_N = 2
