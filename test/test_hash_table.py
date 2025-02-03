import jax
import jax.numpy as jnp
import pytest

from JAxtar.hash import HashTable, hash_func_builder
from JAxtar.util import set_tree
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

    lookup = jax.jit(lambda table, sample: HashTable.lookup(table, hash_func, sample))
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

    lookup = jax.jit(lambda table, sample: HashTable.lookup(table, hash_func, sample))
    parallel_insert = jax.jit(
        lambda table, sample, filled: HashTable.parallel_insert(table, hash_func, sample, filled)
    )

    # Check initial state
    _, _, old_found = jax.vmap(lookup, in_axes=(None, 0))(table, sample)
    assert not jnp.any(old_found)

    # Insert states
    batched_sample, filled = HashTable.make_batched(puzzle.State, sample, batch)
    table, inserted, _, _, _ = parallel_insert(table, batched_sample, filled)

    # Verify insertion
    _, _, found = jax.vmap(lookup, in_axes=(None, 0))(table, sample)
    assert jnp.all(found)  # All states should be found after insertion
    assert jnp.mean(inserted) > 0  # Some states should have been inserted


def test_same_state_insert_at_batch(puzzle, hash_func):
    batch = 5000
    table = HashTable.build(puzzle.State, 1, int(1e5))
    parallel_insert = jax.jit(
        lambda table, sample, filled: HashTable.parallel_insert(table, hash_func, sample, filled)
    )
    lookup = jax.jit(lambda table, sample: HashTable.lookup(table, hash_func, sample))

    num = 10
    counts = 0
    all_samples = []
    for i in range(num):
        key = jax.random.PRNGKey(i)
        samples = jax.vmap(puzzle.get_initial_state)(key=jax.random.split(key, batch))
        cloned_sample_num = jax.random.randint(key, (), 1, batch // 2)
        cloned_sample_idx = jax.random.randint(key, (cloned_sample_num,), 0, batch - 2)
        cloned_sample_idx = jnp.sort(cloned_sample_idx)

        # Create deliberate duplicates within the batch
        samples = set_tree(samples, samples[cloned_sample_idx], cloned_sample_idx + 1)
        h, bytesed = jax.vmap(hash_func, in_axes=(0, None))(samples, 0)
        unique_count = jnp.unique(bytesed, axis=0).shape[0]
        # after this, some states are duplicated
        all_samples.append(samples)

        batched_sample, filled = HashTable.make_batched(puzzle.State, samples, batch)
        table, updatable, unique, idxs, table_idxs = parallel_insert(table, batched_sample, filled)
        counts += jnp.sum(updatable)

        # Verify uniqueness tracking
        unique_idxs = jnp.unique(jnp.stack([idxs, table_idxs], axis=1), axis=0)
        assert (
            unique_idxs.shape[0] == unique_count
        ), f"unique_idxs.shape: {unique_idxs.shape}, unique_count: {unique_count}"
        assert unique_idxs.shape[0] == jnp.sum(unique), "Unique index mismatch"
        assert jnp.all(
            jnp.unique(unique_idxs, axis=0) == unique_idxs
        ), "Duplicate indices in unique set"

        # Verify inserted states exist in table
        _, _, found = jax.vmap(lookup, in_axes=(None, 0))(table, samples)
        assert jnp.all(found), (
            "Inserted states not found in table\n",
            f"unique_count: {unique_count}\n",
            f"unique_idxs.shape: {unique_idxs.shape}, unique: {jnp.sum(unique)}\n",
            f"found: {jnp.sum(found)}\n",
            f"not_found_idxs: {jnp.where(~found)[0]}\n",
            f"cloned_sample_idx: {cloned_sample_idx}\n",
        )

    # Final validation
    assert table.size == counts, f"Size mismatch: {table.size} vs {counts}"

    # Verify cross-batch duplicates
    for samples in all_samples:
        idx, table_idx, found = jax.vmap(lookup, in_axes=(None, 0))(table, samples)
        assert jnp.all(found), "Cross-batch state missing"
        contents = table.table[idx, table_idx]
        assert jnp.all(
            jax.vmap(puzzle.is_equal)(contents, samples)
        ), "Inserted states not found in table"


def test_large_hash_table(puzzle, hash_func):
    count = int(1e7)
    batch = int(1e4)
    table = HashTable.build(puzzle.State, 1, count)

    sample = jax.vmap(puzzle.get_initial_state)(key=jax.random.split(jax.random.PRNGKey(2), count))
    hash, bytes = jax.vmap(hash_func, in_axes=(0, None))(sample, 0)
    unique_bytes = jnp.unique(bytes, axis=0, return_index=True)[1]
    unique_bytes_len = unique_bytes.shape[0]
    unique_hash = jnp.unique(hash, axis=0, return_index=True)[1]
    unique_hash_len = unique_hash.shape[0]
    print(f"unique_bytes_len: {unique_bytes_len}, unique_hash_len: {unique_hash_len}")

    parallel_insert = jax.jit(
        lambda table, sample, filled: HashTable.parallel_insert(table, hash_func, sample, filled)
    )
    lookup = jax.jit(lambda table, sample: HashTable.lookup(table, hash_func, sample))

    # Insert in batches
    inserted_count = 0
    for i in range(0, count, batch):
        batch_sample = sample[i : i + batch]
        table, inserted, _, _, _ = parallel_insert(
            table, batch_sample, jnp.ones(len(batch_sample), dtype=jnp.bool_)
        )
        inserted_count += jnp.sum(inserted)

    assert (
        inserted_count == unique_bytes_len
    ), f"inserted_count: {inserted_count}, unique_bytes_len: {unique_bytes_len}, unique_hash_len: {unique_hash_len}"

    # Verify all states can be found
    _, _, found = jax.vmap(lookup, in_axes=(None, 0))(table, sample)
    assert jnp.mean(found) == 1.0  # All states should be found
