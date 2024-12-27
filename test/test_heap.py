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
    def default(_=None) -> "HeapValue":
        return HeapValue(
            a=jnp.full((), jnp.inf, dtype=jnp.uint8),
            b=jnp.full((1, 2), jnp.inf, dtype=jnp.uint32),
            c=jnp.full((1, 2, 3), jnp.inf, dtype=jnp.float32),
        )


@pytest.fixture
def heap_setup():
    batch_size = 128
    max_size = 1000
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
    value = jax.vmap(HeapValue.default)(jnp.arange(batch_size))

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
    value = jax.vmap(HeapValue.default)(jnp.arange(max_size + 100))

    with pytest.raises(Exception):  # Should raise an exception when exceeding max size
        heap = BGPQ.insert(heap, key, value)


def test_heap_batch_operations(heap_setup):
    heap, batch_size, max_size = heap_setup

    # Test batch insertion
    for i in range(0, 512, batch_size):
        key = jax.random.uniform(
            jax.random.PRNGKey(i), shape=(batch_size,), minval=0, maxval=10, dtype=KEY_DTYPE
        )
        value = jax.vmap(HeapValue.default)(jnp.arange(i, i + batch_size))
        heap = BGPQ.insert(heap, key, value)

    assert heap.size == 512, f"Expected size 512, got {heap.size}"

    # Test batch deletion
    all_mins = []
    while heap.size > 0:
        heap, min_key, min_val = BGPQ.delete_mins(heap)
        all_mins.extend(min_key.tolist())

    # Verify that elements are in ascending order
    assert all(all_mins[i] <= all_mins[i + 1] for i in range(len(all_mins) - 1))
