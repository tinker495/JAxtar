import jax.numpy as jnp

from JAxtar.id_stars.id_frontier import compact_by_valid


def test_compact_by_valid_dense_fast_path_preserves_order():
    values = jnp.array([10, 20, 30, 40], dtype=jnp.int32)
    valid_mask = jnp.array([True, True, True, True])

    packed_values, packed_valid, valid_count, valid_idx = compact_by_valid(values, valid_mask)

    assert packed_values.tolist() == values.tolist()
    assert packed_valid.tolist() == [True, True, True, True]
    assert int(valid_count) == 4
    assert valid_idx.tolist() == [0, 1, 2, 3]


def test_compact_by_valid_sparse_path_matches_original_semantics():
    values = jnp.array([10, 20, 30, 40], dtype=jnp.int32)
    valid_mask = jnp.array([True, False, True, False])

    packed_values, packed_valid, valid_count, valid_idx = compact_by_valid(values, valid_mask)

    assert packed_values.tolist() == [10, 30, 10, 10]
    assert packed_valid.tolist() == [True, True, False, False]
    assert int(valid_count) == 2
    assert valid_idx.tolist() == [0, 2, 0, 0]


def test_compact_by_valid_empty_fast_path_marks_all_invalid():
    values = jnp.array([10, 20, 30, 40], dtype=jnp.int32)
    valid_mask = jnp.array([False, False, False, False])

    packed_values, packed_valid, valid_count, valid_idx = compact_by_valid(values, valid_mask)

    assert packed_values.tolist() == values.tolist()
    assert packed_valid.tolist() == [False, False, False, False]
    assert int(valid_count) == 0
    assert valid_idx.tolist() == [0, 0, 0, 0]
