import os

import jax.numpy as jnp

KEY_DTYPE = jnp.float16
ACTION_DTYPE = jnp.uint8
HASH_SIZE_MULTIPLIER = int(os.environ.get("JAXTAR_HASH_SIZE_MULTIPLIER", "2"))
CUCKOO_TABLE_N = int(os.environ.get("JAXTAR_HASH_BUCKET_SIZE", "16"))
MIN_BATCH_SIZE = 128
