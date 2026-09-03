import jax.numpy as jnp
import pytest
from xtructure import HashIdx

from JAxtar.stars.astar import _row_heuristics
from JAxtar.stars.search_base import Current
from JAxtar.utils.batch_switcher import variable_batch_switcher_builder

ACTIONS, BATCH, MIN_BATCH = 3, 8, 2
NN_VALUE = 7.0


def _fake_heuristic(params, states):
    return jnp.full((states.shape[0],), NN_VALUE, dtype=jnp.float32)


def _setup(n_new: int):
    switcher = variable_batch_switcher_builder(
        _fake_heuristic, max_batch_size=BATCH, min_batch_size=MIN_BATCH, pad_value=jnp.inf
    )
    flat = ACTIONS * BATCH
    neighbours = jnp.arange(flat, dtype=jnp.float32).reshape(ACTIONS, BATCH)
    index = jnp.arange(flat, dtype=jnp.uint32).reshape(ACTIONS, BATCH)
    vals = Current(hashidx=HashIdx(index=index), cost=jnp.zeros((ACTIONS, BATCH), jnp.float16))
    new_mask = (jnp.arange(flat) < n_new).reshape(ACTIONS, BATCH)
    dist = (100.0 + jnp.arange(flat, dtype=jnp.float32)).astype(jnp.float16)
    return switcher, neighbours, vals, new_mask, dist


def _run(n_new: int):
    switcher, neighbours, vals, new_mask, dist = _setup(n_new)
    new_dist, heurs = _row_heuristics(
        _fake_heuristic, switcher, None, neighbours, vals, new_mask, dist, BATCH, ACTIONS
    )
    cached = dist[vals.hashidx.index]
    return new_dist, heurs, cached


@pytest.mark.parametrize("n_new", [0, BATCH, 2 * BATCH, ACTIONS * BATCH])
def test_rows_without_new_states_keep_cached_values(n_new):
    """n_new is a multiple of the batch: there is no mixed row, so every row past the
    all-new prefix must return the cached values (an all-False switcher mask would
    evaluate MIN_BATCH entries and pad the rest with inf)."""
    new_dist, heurs, cached = _run(n_new)
    n_full = n_new // BATCH
    for row in range(ACTIONS):
        expected = jnp.full((BATCH,), NN_VALUE) if row < n_full else cached[row]
        assert jnp.array_equal(heurs[row], expected.astype(heurs.dtype)), row
    untouched = new_dist[n_new:] if n_new < ACTIONS * BATCH else new_dist[:0]
    assert jnp.array_equal(untouched, dist_slice(n_new))


def dist_slice(n_new):
    _, _, _, _, dist = _setup(n_new)
    return dist[n_new:] if n_new < ACTIONS * BATCH else dist[:0]


def test_mixed_row_keeps_switcher_padding():
    """A genuinely mixed row is evaluated through the switcher: values for the evaluated
    slice, inf beyond it (existing behaviour), and rows after it stay cached."""
    n_new = BATCH + 3  # row 0 full, row 1 has 3 new states, row 2 none
    _, heurs, cached = _run(n_new)
    assert jnp.array_equal(heurs[0], jnp.full((BATCH,), NN_VALUE, dtype=heurs.dtype))
    evaluated = 4  # smallest switcher branch >= 3 new states
    assert jnp.array_equal(heurs[1][:evaluated], jnp.full((evaluated,), NN_VALUE, heurs.dtype))
    assert bool(jnp.all(jnp.isinf(heurs[1][evaluated:])))
    assert jnp.array_equal(heurs[2], cached[2])
