from functools import partial

import jax
import jax.numpy as jnp
import pytest

from JAxtar.hash import HashTable, hash_func_builder
from puzzle.slidepuzzle import SlidePuzzle


@pytest.fixture
def puzzle():
    return SlidePuzzle(4)


@pytest.fixture
def hash_func(puzzle):
    return hash_func_builder(puzzle.State)


def test_hash_table_lookup(puzzle, hash_func):
    count = 1000
    sample = jax.vmap(puzzle.get_initial_state)(key=jax.random.split(jax.random.PRNGKey(2), count))
    table = HashTable.build(puzzle.State, 1, int(1e4))

    lookup = jax.jit(partial(HashTable.lookup, hash_func))
    idx, table_idx, found = jax.vmap(lookup, in_axes=(None, 0))(table, sample)

    assert idx.shape == (count,)
    assert table_idx.shape == (count,)
    assert found.shape == (count,)
    assert not jnp.any(found)  # Initially all should be not found


def test_hash_table_insert(puzzle, hash_func):
    count = 1000
    batch = 4000
    table = HashTable.build(puzzle.State, 1, int(1e4))

    sample = jax.vmap(puzzle.get_initial_state)(
        key=jax.random.split(jax.random.PRNGKey(256), count)
    )

    lookup = jax.jit(partial(HashTable.lookup, hash_func))
    parallel_insert = jax.jit(partial(HashTable.parallel_insert, hash_func))

    # Check initial state
    _, _, old_found = jax.vmap(lookup, in_axes=(None, 0))(table, sample)
    assert not jnp.any(old_found)

    # Insert states
    batched_sample, filled = HashTable.make_batched(puzzle.State, sample, batch)
    table, inserted, _, _ = parallel_insert(table, batched_sample, filled)

    # Verify insertion
    _, _, found = jax.vmap(lookup, in_axes=(None, 0))(table, sample)
    assert jnp.all(found)  # All states should be found after insertion
    assert jnp.mean(inserted) > 0  # Some states should have been inserted


def test_same_state_insert_at_batch(puzzle, hash_func):
    count = 1000
    batch = 5000
    table = HashTable.build(puzzle.State, 1, int(1e5))
    parallel_insert = jax.jit(partial(HashTable.parallel_insert, hash_func))

    num = 10
    counts = 0
    for i in range(num):
        sample = puzzle.get_initial_state(jax.random.PRNGKey(i))
        sample = jax.tree_util.tree_map(lambda x: jnp.repeat(x[None], count, axis=0), sample)

        batched_sample, filled = HashTable.make_batched(puzzle.State, sample, batch)
        table, _, idxs, table_idxs = parallel_insert(table, batched_sample, filled)
        unique_idxs = jnp.unique(jnp.stack([idxs, table_idxs], axis=1), axis=0)
        assert unique_idxs.shape[0] == 2, f"unique_idxs.shape: {unique_idxs.shape}"
        counts += 1

    assert table.size == counts, f"table.table.size: {table.size}, counts: {counts}"


def test_large_hash_table(puzzle, hash_func):
    count = int(1e7)
    batch = int(1e4)
    table = HashTable.build(puzzle.State, 1, count)

    sample = jax.vmap(puzzle.get_initial_state)(key=jax.random.split(jax.random.PRNGKey(2), count))
    unique_sample = jnp.unique(sample.get_flatten(), axis=0)
    unique_len = unique_sample.shape[0]

    parallel_insert = jax.jit(partial(HashTable.parallel_insert, hash_func))
    lookup = jax.jit(partial(HashTable.lookup, hash_func))

    # Insert in batches
    inserted_count = 0
    for i in range(0, count, batch):
        batch_sample = sample[i : i + batch]
        table, inserted, _, _ = parallel_insert(
            table, batch_sample, jnp.ones(len(batch_sample), dtype=jnp.bool_)
        )
        inserted_count += jnp.sum(inserted)

    assert inserted_count == unique_len  #

    # Verify all states can be found
    _, _, found = jax.vmap(lookup, in_axes=(None, 0))(table, sample)
    assert jnp.mean(found) == 1.0  # All states should be found
