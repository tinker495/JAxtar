"""Tests for batch_switcher optimization with auto-splitting and Pytree support."""

from typing import NamedTuple

import jax
import jax.numpy as jnp

from JAxtar.utils.batch_switcher import (
    build_batch_sizes_for_cap,
    prefix_batch_switcher_builder,
    variable_batch_switcher_builder,
)


class MockState(NamedTuple):
    data: jnp.ndarray
    extra: jnp.ndarray


def test_variable_batch_switcher():
    """Test variable_batch_switcher for various batch sizes and Pytree inputs."""

    def eval_fn(params, state):
        # state is MockState, returns (batch, 1)
        return jnp.sum(state.data, axis=-1, keepdims=True) + params

    # 1. Small Batch (uses default min_batch_size)
    switcher = variable_batch_switcher_builder(
        eval_fn, pad_value=0.0, expected_output_shape=(1,), expected_output_dtype=jnp.float32
    )

    B = 32
    params = 1.0
    state = MockState(data=jnp.ones((B, 10)), extra=jnp.zeros((B,)))
    filled = jnp.ones((B,), dtype=bool)

    res = switcher(params, state, filled)
    assert res.shape == (B, 1), f"Expected ({B}, 1), got {res.shape}"
    assert jnp.allclose(res, 11.0), f"Expected 11.0, got {res[0]}"

    # 2. Large Batch (Auto-splitting)
    switcher_large = variable_batch_switcher_builder(eval_fn, pad_value=0.0)

    N = 5000
    state_large = MockState(data=jnp.ones((N, 10)), extra=jnp.zeros((N,)))
    filled_large = jnp.ones((N,), dtype=bool)

    jit_switcher = jax.jit(switcher_large)
    res_large = jit_switcher(params, state_large, filled_large)
    assert res_large.shape == (N, 1), f"Expected ({N}, 1), got {res_large.shape}"
    assert jnp.allclose(res_large, 11.0)

    # 3. Zero Batch (min_batch_size=0)
    switcher_zero = variable_batch_switcher_builder(
        eval_fn,
        pad_value=0.0,
        min_batch_size=0,
        expected_output_shape=(1,),
        expected_output_dtype=jnp.float32,
    )

    state_zero = MockState(data=jnp.zeros((0, 10)), extra=jnp.zeros((0,)))
    filled_zero = jnp.zeros((0,), dtype=bool)

    res_zero = switcher_zero(params, state_zero, filled_zero)
    assert res_zero.shape == (0, 1), f"Expected (0, 1), got {res_zero.shape}"

    # 4. Partial Validity
    B_partial = 512
    state_partial = MockState(data=jnp.ones((B_partial, 10)), extra=jnp.zeros((B_partial,)))
    filled_partial = jnp.zeros((B_partial,), dtype=bool).at[:10].set(True)

    switcher_partial = variable_batch_switcher_builder(
        eval_fn, pad_value=0.0, expected_output_shape=(1,), expected_output_dtype=jnp.float32
    )

    res_partial = switcher_partial(params, state_partial, filled_partial)
    assert res_partial.shape == (B_partial, 1)
    assert jnp.allclose(res_partial[:10], 11.0)


def test_non_batched_leaf_raises():
    """Test that non-batched leaves raise ValueError (fail-fast contract)."""

    class BadState(NamedTuple):
        data: jnp.ndarray
        scalar_metadata: jnp.ndarray  # This will be 0-dim, which is invalid

    def eval_fn(params, state):
        return jnp.sum(state.data, axis=-1, keepdims=True)

    switcher = variable_batch_switcher_builder(eval_fn, pad_value=0.0)

    B = 32
    # Create a state with a scalar leaf (0-dim) - this should fail
    bad_state = BadState(data=jnp.ones((B, 10)), scalar_metadata=jnp.array(1.0))
    filled = jnp.ones((B,), dtype=bool)

    try:
        switcher(1.0, bad_state, filled)
        assert False, "Expected ValueError for non-batched leaf"
    except ValueError as e:
        assert "leading batch dims" in str(e)


def test_build_batch_sizes_for_cap():
    sizes = build_batch_sizes_for_cap(1024, min_batch_unit=128)
    assert sizes == [128, 256, 512, 1024]

    sizes_small = build_batch_sizes_for_cap(100, min_batch_unit=128)
    assert sizes_small == [100]


def test_all_invalid_returns_pad_values():
    def eval_fn(params, state):
        return jnp.sum(state.data, axis=-1, keepdims=True) + params

    switcher = variable_batch_switcher_builder(
        eval_fn,
        pad_value=-9.0,
        batch_sizes=[8, 16],
        partition_mode="flat",
        expected_output_shape=(1,),
        expected_output_dtype=jnp.float32,
    )

    state = MockState(
        data=jnp.arange(8 * 4, dtype=jnp.float32).reshape(8, 4), extra=jnp.zeros((8,))
    )
    filled = jnp.zeros((8,), dtype=jnp.bool_)
    out = jax.jit(switcher)(3.0, state, filled)
    assert out.shape == (8, 1)
    assert jnp.allclose(out, -9.0)


def test_assume_prefix_packed_matches_default_for_prefix_mask():
    def eval_fn(params, state):
        return jnp.sum(state.data, axis=-1, keepdims=True) + params

    base_switcher = variable_batch_switcher_builder(
        eval_fn,
        pad_value=-3.0,
        batch_sizes=[2, 4, 8],
        partition_mode="flat",
    )
    fast_switcher = variable_batch_switcher_builder(
        eval_fn,
        pad_value=-3.0,
        batch_sizes=[2, 4, 8],
        partition_mode="flat",
        assume_prefix_packed=True,
    )

    state = MockState(
        data=jnp.arange(8 * 3, dtype=jnp.float32).reshape(8, 3),
        extra=jnp.zeros((8,), dtype=jnp.float32),
    )
    filled = jnp.zeros((8,), dtype=jnp.bool_).at[:5].set(True)

    out_base = jax.jit(base_switcher)(1.5, state, filled)
    out_fast = jax.jit(fast_switcher)(1.5, state, filled)
    assert out_base.shape == out_fast.shape == (8, 1)
    assert jnp.allclose(out_fast[filled], out_base[filled])


def test_max_batch_size_override_handles_split_input():
    def eval_fn(params, state):
        return jnp.sum(state.data, axis=-1, keepdims=True) + params

    switcher = variable_batch_switcher_builder(
        eval_fn,
        pad_value=-1.0,
        max_batch_size=16,
        batch_sizes=[4, 8, 16],
        partition_mode="flat",
    )

    n = 32
    state = MockState(
        data=jnp.arange(n * 2, dtype=jnp.float32).reshape(n, 2),
        extra=jnp.zeros((n,), dtype=jnp.float32),
    )
    valid_indices = jnp.array([1, 5, 17, 20], dtype=jnp.int32)
    filled = jnp.zeros((n,), dtype=jnp.bool_).at[valid_indices].set(True)

    out = jax.jit(switcher)(2.0, state, filled)
    expected = eval_fn(2.0, state)
    assert out.shape == (n, 1)
    assert jnp.allclose(out[valid_indices], expected[valid_indices])


def test_prefix_batch_switcher_builder_prefix_mask():
    def eval_fn(params, state):
        return jnp.sum(state, axis=-1, keepdims=True) + params

    switcher = prefix_batch_switcher_builder(
        eval_fn,
        max_batch_size=16,
        min_batch_size=4,
        pad_value=-1.0,
    )

    state = jnp.arange(16 * 2, dtype=jnp.float32).reshape(16, 2)
    filled = jnp.zeros((16,), dtype=jnp.bool_).at[:6].set(True)

    out = jax.jit(switcher)(1.0, state, filled)
    expected = eval_fn(1.0, state)
    assert out.shape == (16, 1)
    assert jnp.allclose(out[:6], expected[:6])


if __name__ == "__main__":
    test_variable_batch_switcher()
    test_non_batched_leaf_raises()
    test_build_batch_sizes_for_cap()
    test_all_invalid_returns_pad_values()
    test_assume_prefix_packed_matches_default_for_prefix_mask()
    test_max_batch_size_override_handles_split_input()
    test_prefix_batch_switcher_builder_prefix_mask()
    print("All tests passed!")
