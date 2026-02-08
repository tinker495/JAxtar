import jax.numpy as jnp
import pytest
from xtructure import HashIdx

from JAxtar.stars.search_base import (
    Current,
    _compute_pop_process_mask,
    _unique_sort_merge_and_split,
)


def _build_current(indices: list[int], costs: list[float]) -> Current:
    return Current(
        hashidx=HashIdx(index=jnp.array(indices, dtype=jnp.uint32)),
        cost=jnp.array(costs, dtype=jnp.float32),
    )


def test_unique_sort_merge_and_split_prefers_lower_cost_duplicate_representative():
    k1 = jnp.array([1.0, 1.1, 2.0, jnp.inf], dtype=jnp.float32)
    k2 = jnp.array([1.05, 1.2, 1.3, jnp.inf], dtype=jnp.float32)
    v1 = _build_current([10, 20, 10, 99], [5.0, 1.0, 3.0, 9.0])
    v2 = _build_current([30, 20, 40, 88], [1.0, 0.5, 2.0, 9.0])

    main_key, main_val, overflow_key, _ = _unique_sort_merge_and_split(k1, v1, k2, v2, batch_size=4)

    assert main_key.tolist() == pytest.approx([1.05, 1.2, 1.3, 2.0])
    assert main_val.hashidx.index.tolist() == [30, 20, 40, 10]
    assert main_val.cost.tolist() == pytest.approx([1.0, 0.5, 2.0, 3.0])
    assert overflow_key.tolist() == pytest.approx([jnp.inf, jnp.inf, jnp.inf, jnp.inf])


def test_deferred_selection_is_based_on_final_key_batch():
    final_key = jnp.array([1.0, 1.05, 2.0, jnp.inf], dtype=jnp.float32)
    initial_min_key = jnp.array([jnp.inf, jnp.inf, jnp.inf, jnp.inf], dtype=jnp.float32)

    new_mask = _compute_pop_process_mask(final_key, pop_ratio=1.01, min_pop=1)

    # Regression baseline: old deferred logic mistakenly computed the mask from `min_key`.
    filled = jnp.isfinite(final_key)
    old_threshold = initial_min_key[0] * 1.01 + 1e-6
    old_process_mask = jnp.less_equal(initial_min_key, old_threshold)
    old_base_mask = jnp.logical_and(filled, old_process_mask)
    old_min_pop_mask = jnp.logical_and(jnp.cumsum(filled) <= 1, filled)
    old_mask = jnp.logical_or(old_base_mask, old_min_pop_mask)

    assert new_mask.tolist() == [True, False, False, False]
    assert old_mask.tolist() == [True, True, True, False]
