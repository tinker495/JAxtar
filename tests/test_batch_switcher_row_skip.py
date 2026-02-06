"""Tests for action-row partition scan and empty-row skipping."""

import jax
import jax.numpy as jnp

from JAxtar.utils.batch_switcher import variable_batch_switcher_builder


def _eval_fn(params, state):
    return jnp.sum(state, axis=-1, keepdims=True) + params


def test_row_scan_skips_empty_rows_and_preserves_valid_values():
    switcher = variable_batch_switcher_builder(
        _eval_fn,
        pad_value=-7.0,
        batch_sizes=[1, 2, 4, 8],
        partition_mode="row_scan",
        expected_output_shape=(1,),
        expected_output_dtype=jnp.float32,
    )

    state = jnp.arange(3 * 8 * 5, dtype=jnp.float32).reshape(3, 8, 5)
    filled = jnp.zeros((3, 8), dtype=jnp.bool_)
    filled = filled.at[1, :2].set(True)
    filled = filled.at[2, 4].set(True)

    out = jax.jit(switcher)(1.0, state, filled)
    assert out.shape == (3, 8, 1)

    # Empty rows should be returned as pure pad without calling eval_fn.
    assert jnp.allclose(out[0], -7.0)

    expected = _eval_fn(1.0, state)
    assert jnp.allclose(out[1, :2], expected[1, :2])
    assert jnp.allclose(out[2, 4], expected[2, 4])


def test_auto_partition_matches_flat_on_valid_entries():
    auto_switcher = variable_batch_switcher_builder(
        _eval_fn,
        pad_value=-5.0,
        batch_sizes=[1, 2, 4, 8, 16],
        partition_mode="auto",
        expected_output_shape=(1,),
        expected_output_dtype=jnp.float32,
    )
    flat_switcher = variable_batch_switcher_builder(
        _eval_fn,
        pad_value=-5.0,
        batch_sizes=[1, 2, 4, 8, 16],
        partition_mode="flat",
        expected_output_shape=(1,),
        expected_output_dtype=jnp.float32,
    )

    state = jnp.ones((4, 16, 6), dtype=jnp.float32) * 3.0
    filled = jnp.zeros((4, 16), dtype=jnp.bool_)
    filled = filled.at[0, :3].set(True)
    filled = filled.at[2, :5].set(True)
    filled = filled.at[3, 7:9].set(True)

    out_auto = jax.jit(auto_switcher)(2.0, state, filled)
    out_flat = jax.jit(flat_switcher)(2.0, state, filled)

    valid_mask = filled[..., None]
    assert out_auto.shape == out_flat.shape == (4, 16, 1)
    assert jnp.allclose(jnp.where(valid_mask, out_auto, 0.0), jnp.where(valid_mask, out_flat, 0.0))

    # Row 1 is fully empty and should become pure pad in auto(row_scan) mode.
    assert jnp.allclose(out_auto[1], -5.0)
