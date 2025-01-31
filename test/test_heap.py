import chex
import jax
import jax.numpy as jnp
import pytest

from JAxtar.annotate import KEY_DTYPE
from JAxtar.bgpq import BGPQ, bgpq_value_dataclass


@bgpq_value_dataclass
class HeapValue:
    """
    This class is a dataclass that represents a hash table heap value.
    It has two fields:
    1. index: hashtable index
    2. table_index: cuckoo table index
    """

    a: chex.Array
    b: chex.Array
    c: chex.Array

    @staticmethod
    def default(shape=()) -> "HeapValue":
        return HeapValue(
            a=jnp.full(shape, jnp.inf, dtype=jnp.uint8),
            b=jnp.full(shape + (1, 2), jnp.inf, dtype=jnp.uint32),
            c=jnp.full(shape + (1, 2, 3), jnp.inf, dtype=jnp.float32),
        )

    def random(shape=(), key=None):
        if key is None:
            key = jax.random.PRNGKey(0)
        key1, key2, key3 = jax.random.split(key, 3)
        return HeapValue(
            a=jax.random.randint(
                key1,
                shape=shape,
                minval=0,
                maxval=10,
            ).astype(jnp.uint8),
            b=jax.random.randint(
                key2,
                shape=shape + (1, 2),
                minval=0,
                maxval=10,
            ).astype(jnp.uint32),
            c=jax.random.uniform(
                key3,
                shape=shape + (1, 2, 3),
            ).astype(jnp.float32),
        )


def rotl(x, n):
    """Rotate left operation for 32-bit integers."""
    return (x << n) | (x >> (32 - n))


@jax.jit
def xxhash(x, seed):
    """
    Implementation of xxHash algorithm for 32-bit integers.
    Args:
        x: Input value to hash
        seed: Seed value for hash function
    Returns:
        32-bit hash value
    """
    prime_1 = jnp.uint32(0x9E3779B1)
    prime_2 = jnp.uint32(0x85EBCA77)
    prime_3 = jnp.uint32(0xC2B2AE3D)
    prime_5 = jnp.uint32(0x165667B1)
    acc = jnp.uint32(seed) + prime_5
    for _ in range(4):
        lane = x & 255
        acc = acc + lane * prime_5
        acc = rotl(acc, 11) * prime_1
        x = x >> 8
    acc = acc ^ (acc >> 15)
    acc = acc * prime_2
    acc = acc ^ (acc >> 13)
    acc = acc * prime_3
    acc = acc ^ (acc >> 16)
    return acc


def heap_key_builder(x: HeapValue):
    @jax.jit
    def _to_bytes(x):
        """Convert input to byte array."""
        return jax.lax.bitcast_convert_type(x, jnp.uint8).reshape(-1)

    @jax.jit
    def _byterize(x):
        """Convert entire state tree to flattened byte array."""
        x = jax.tree_util.tree_map(_to_bytes, x)
        x, _ = jax.tree_util.tree_flatten(x)
        return jnp.concatenate(x)

    default_bytes = _byterize(x.default())
    bytes_len = default_bytes.shape[0]
    # Calculate padding needed to make byte length multiple of 4
    pad_len = jnp.where(bytes_len % 4 != 0, 4 - (bytes_len % 4), 0)

    if pad_len > 0:

        def _to_uint32s(bytes):
            """Convert padded bytes to uint32 array."""
            x_padded = jnp.pad(bytes, (pad_len, 0), mode="constant", constant_values=0)
            x_reshaped = jnp.reshape(x_padded, (-1, 4))
            return jax.vmap(lambda x: jax.lax.bitcast_convert_type(x, jnp.uint32))(
                x_reshaped
            ).reshape(-1)

    else:

        def _to_uint32s(bytes):
            """Convert bytes directly to uint32 array."""
            x_reshaped = jnp.reshape(bytes, (-1, 4))
            return jax.vmap(lambda x: jax.lax.bitcast_convert_type(x, jnp.uint32))(
                x_reshaped
            ).reshape(-1)

    def _keys(x):
        bytes = _byterize(x)
        uint32ed = _to_uint32s(bytes)

        def scan_body(seed, x):
            result = xxhash(x, seed)
            return result, result

        hash_value, _ = jax.lax.scan(scan_body, 1, uint32ed)
        hash_value = (hash_value % (2**12)) / (2**8)
        return hash_value.astype(KEY_DTYPE)

    return jax.jit(_keys)


@pytest.fixture
def heap_setup():
    batch_size = 128
    max_size = 100000
    heap = BGPQ.build(max_size, batch_size, HeapValue)
    return heap, batch_size, max_size


def test_heap_initialization(heap_setup):
    heap, batch_size, max_size = heap_setup
    assert heap is not None
    assert heap.size == 0
    assert heap.batch_size == batch_size


def test_heap_insert_and_delete(heap_setup):
    heap, batch_size, max_size = heap_setup

    # Test inserting elements
    key = jax.random.uniform(
        jax.random.PRNGKey(0), shape=(batch_size,), minval=0, maxval=10, dtype=KEY_DTYPE
    )
    value = HeapValue.random(shape=(batch_size,), key=jax.random.PRNGKey(0))

    # Insert elements
    heap = BGPQ.insert(heap, key, value)
    assert heap.size == 128

    # Test deleting elements
    last_min = float("-inf")
    while heap.size > 0:
        heap, min_key, min_val = BGPQ.delete_mins(heap)
        current_min = jnp.min(min_key)
        assert current_min >= last_min  # Check if elements are coming out in sorted order
        last_min = current_min


def test_heap_overflow(heap_setup):
    heap, batch_size, max_size = heap_setup

    # Try to insert more elements than max_size
    key = jax.random.uniform(
        jax.random.PRNGKey(0), shape=(max_size + 100,), minval=0, maxval=10, dtype=KEY_DTYPE
    )
    value = HeapValue.random(shape=(max_size + 100,), key=jax.random.PRNGKey(0))

    with pytest.raises(Exception):  # Should raise an exception when exceeding max size
        heap = BGPQ.insert(heap, key, value)


def test_heap_batch_operations(heap_setup):
    heap, batch_size, max_size = heap_setup

    _key_gen = heap_key_builder(HeapValue)
    _key_gen = jax.jit(jax.vmap(_key_gen))

    # Test batch insertion
    for i in range(0, 512, 1):

        value = HeapValue.random(shape=(batch_size,), key=jax.random.PRNGKey(i))
        key = _key_gen(value)
        heap = BGPQ.insert(heap, key, value)

    assert heap.size == 512 * batch_size, f"Expected size 512 * batch_size, got {heap.size}"

    # Test batch deletion
    all_mins = []
    while heap.size > 0:
        heap, min_key, min_val = BGPQ.delete_mins(heap)

        # check key and value matching
        isclose = jnp.isclose(min_key, _key_gen(min_val))
        # TODO: this is not passed, must be fixed
        assert jnp.all(isclose), (
            f"Key and value mismatch, \nmin_key: \n{min_key},"
            f"\nmin_val_key: \n{_key_gen(min_val)},"
            f"\nidexs: \n{jnp.where(~isclose)}"
        )
        all_mins.extend(min_key.tolist())

    # Verify that elements are in ascending order
    assert all(all_mins[i] <= all_mins[i + 1] for i in range(len(all_mins) - 1))
