"""Tests for batch_switcher optimization with auto-splitting and Pytree support."""

from typing import NamedTuple

import jax
import jax.numpy as jnp

from JAxtar.utils.batch_switcher import variable_batch_switcher_builder


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


if __name__ == "__main__":
    test_variable_batch_switcher()
    test_non_batched_leaf_raises()
    print("All tests passed!")
