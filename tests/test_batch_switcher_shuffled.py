"""Tests for batch_switcher optimization with shuffled inputs."""
from typing import NamedTuple

import jax
import jax.numpy as jnp

from JAxtar.utils.batch_switcher import variable_batch_switcher_builder


class MockState(NamedTuple):
    data: jnp.ndarray


def test_shuffled_validity():
    def eval_fn(params, state):
        return jnp.sum(state.data, axis=-1, keepdims=True) + params

    # Partial validity with shuffling
    # limit_batch_size is default 4096 (SPLIT_UNIT).
    # Let's verify with smaller Limit to trigger chunking or just standard?
    # batch_switcher defaults to BATCH_SPLIT_UNIT=4096.

    B = 256
    # pad_value = -1.0 to distinguish from zeros
    switcher = variable_batch_switcher_builder(eval_fn, pad_value=-1.0)

    # Valid items at indices [10, 20, 30]
    valid_indices = jnp.array([10, 20, 30])
    filled = jnp.zeros((B,), dtype=bool).at[valid_indices].set(True)

    state = MockState(data=jnp.ones((B, 10)) * 2)  # Value 2
    params = 1.0

    # Run
    # JIT to ensure XLA compilation works
    jit_switcher = jax.jit(switcher)
    res = jit_switcher(params, state, filled)

    assert res.shape == (B, 1)

    # Check valid: sum(2)*10 + 1 = 21.0
    assert jnp.allclose(res[valid_indices], 21.0)

    # Invalid items may contain computed garbage or pad_value depending on batch bucket.
    # The caller is responsible for masking them using `filled`.
    # So we do not assert values for invalid_mask.


if __name__ == "__main__":
    test_shuffled_validity()
    print("Shuffled test passed!")
