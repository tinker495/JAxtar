import os

import jax.numpy as jnp

# Search state and loop carries use KEY_DTYPE; cast Puzzle float32 costs at those boundaries.
KEY_DTYPE = jnp.float16
ACTION_DTYPE = jnp.uint8
ACTION_PAD = jnp.array(jnp.iinfo(ACTION_DTYPE).max, dtype=ACTION_DTYPE)
HASH_SIZE_MULTIPLIER = int(os.environ.get("JAXTAR_HASH_SIZE_MULTIPLIER", "1"))
CUCKOO_TABLE_N = int(os.environ.get("JAXTAR_HASH_BUCKET_SIZE", "16"))
DEDUPE_MODE = os.environ.get(
    "JAXTAR_DEDUPE_MODE", os.environ.get("XTRUCTURE_HASHTABLE_DEDUPE_MODE", "approx")
)
os.environ["XTRUCTURE_HASHTABLE_DEDUPE_MODE"] = DEDUPE_MODE
MIN_BATCH_SIZE = 128

# Parent-pointer trace arena (id_stars). TRACE_INVALID marks a chain root and doubles as
# an out-of-bounds scatter target, so `mode="drop"` discards writes past the arena.
TRACE_INDEX_DTYPE = jnp.uint32
TRACE_INVALID_INT = int(jnp.iinfo(TRACE_INDEX_DTYPE).max)
TRACE_INVALID = jnp.array(TRACE_INVALID_INT, dtype=TRACE_INDEX_DTYPE)
