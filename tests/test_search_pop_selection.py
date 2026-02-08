import jax.numpy as jnp

from JAxtar.stars.search_base import _compute_pop_process_mask


def test_compute_pop_process_mask_respects_ratio_window():
    key = jnp.array([1.0, 1.01, 1.2, jnp.inf], dtype=jnp.float32)
    mask = _compute_pop_process_mask(key, pop_ratio=1.05, min_pop=1)
    assert mask.tolist() == [True, True, False, False]


def test_compute_pop_process_mask_enforces_min_pop():
    key = jnp.array([1.0, 10.0, 11.0, jnp.inf], dtype=jnp.float32)
    mask = _compute_pop_process_mask(key, pop_ratio=1.01, min_pop=2)
    assert mask.tolist() == [True, True, False, False]


def test_compute_pop_process_mask_handles_empty_batch():
    key = jnp.array([jnp.inf, jnp.inf], dtype=jnp.float32)
    mask = _compute_pop_process_mask(key, pop_ratio=1.1, min_pop=2)
    assert mask.tolist() == [False, False]
