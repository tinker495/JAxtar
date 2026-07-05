"""Reference-equivalence tests for the shared chunked masked-eval primitive.

``chunked_masked_eval`` is the deep module extracted from the four deferred-search
evaluators. In production ``flat_states`` is a ``Puzzle.State`` handled via
``xnp.take`` / ``xnp.reshape``; here we exercise the partition -> scan -> scatter
logic with plain float arrays and injected value functions, which is sufficient to
pin the clamp, the all-False-chunk skip, and the inverse-permutation scatter.
"""

import jax
import jax.numpy as jnp
import pytest

from JAxtar.annotate import KEY_DTYPE
from JAxtar.utils.chunked_eval import chunked_masked_eval


def _reference(value_fn, flat_states, flat_valid):
    """Naive spec: max(0, value_fn(state)) where valid, inf otherwise."""
    all_true = jnp.ones_like(flat_valid)
    vals = jnp.maximum(0.0, value_fn(flat_states, all_true).astype(KEY_DTYPE))
    return jnp.where(flat_valid, vals, jnp.inf).astype(KEY_DTYPE)


# Two distinct value functions prove the primitive is generic over value_fn
# (the fn is injected, not hard-coded) and that both callable args are honoured.
_VALUE_FNS = [
    lambda s, m: s,
    lambda s, m: 2.0 * s - 3.0,
]


@pytest.mark.parametrize("value_fn", _VALUE_FNS)
def test_matches_reference_random(value_fn):
    action_size, batch_size = 4, 8
    flat_size = action_size * batch_size
    k1, k2 = jax.random.split(jax.random.PRNGKey(0))
    # Integer-valued states keep the float16 comparison exact.
    flat_states = jax.random.randint(k1, (flat_size,), -8, 9).astype(jnp.float32)
    flat_valid = jax.random.bernoulli(k2, 0.5, (flat_size,))

    out = chunked_masked_eval(value_fn, flat_states, flat_valid, action_size, batch_size)

    assert out.shape == (flat_size,)
    assert out.dtype == KEY_DTYPE
    assert jnp.array_equal(out, _reference(value_fn, flat_states, flat_valid))


def test_all_false_chunk_is_all_inf():
    action_size, batch_size = 4, 8
    flat_size = action_size * batch_size
    flat_states = jnp.arange(flat_size, dtype=jnp.float32)
    # Exactly one chunk of valids: after the valid-first partition the remaining
    # three chunks are entirely masked and must hit the cond skip -> all inf.
    flat_valid = jnp.arange(flat_size) < batch_size

    out = chunked_masked_eval(lambda s, m: s, flat_states, flat_valid, action_size, batch_size)

    assert bool(jnp.all(jnp.isinf(out[~flat_valid])))
    assert bool(jnp.all(jnp.isfinite(out[flat_valid])))
    assert jnp.array_equal(out, _reference(lambda s, m: s, flat_states, flat_valid))


def test_no_valid_states_all_inf():
    action_size, batch_size = 3, 4
    flat_size = action_size * batch_size
    flat_states = jnp.arange(flat_size, dtype=jnp.float32)
    flat_valid = jnp.zeros((flat_size,), dtype=jnp.bool_)

    out = chunked_masked_eval(
        lambda s, m: s + 1.0, flat_states, flat_valid, action_size, batch_size
    )

    assert bool(jnp.all(jnp.isinf(out)))


def test_scatter_is_inverse_permutation():
    # Identity value_fn, all valid, distinct values spanning negatives: every value
    # must return to its own index (clamped), proving the scatter inverts the partition.
    action_size, batch_size = 5, 7
    flat_size = action_size * batch_size
    flat_states = jnp.arange(flat_size, dtype=jnp.float32) - 10.0
    flat_valid = jnp.ones((flat_size,), dtype=jnp.bool_)

    out = chunked_masked_eval(lambda s, m: s, flat_states, flat_valid, action_size, batch_size)

    assert jnp.array_equal(out, jnp.maximum(0.0, flat_states).astype(KEY_DTYPE))


def test_mixed_mask_places_values_at_correct_indices():
    action_size, batch_size = 4, 4
    flat_size = action_size * batch_size
    flat_states = jnp.arange(flat_size, dtype=jnp.float32)
    # Valids scattered so they do not align to chunk boundaries.
    flat_valid = jnp.arange(flat_size) % 3 == 0

    out = chunked_masked_eval(lambda s, m: s, flat_states, flat_valid, action_size, batch_size)

    for i in range(flat_size):
        if bool(flat_valid[i]):
            assert float(out[i]) == float(flat_states[i])
        else:
            assert bool(jnp.isinf(out[i]))


def test_clamp_makes_negative_values_zero():
    action_size, batch_size = 2, 4
    flat_size = action_size * batch_size
    flat_states = -5.0 * jnp.ones((flat_size,), dtype=jnp.float32)
    flat_valid = jnp.ones((flat_size,), dtype=jnp.bool_)

    out = chunked_masked_eval(lambda s, m: s, flat_states, flat_valid, action_size, batch_size)

    assert jnp.array_equal(out, jnp.zeros((flat_size,), dtype=KEY_DTYPE))
