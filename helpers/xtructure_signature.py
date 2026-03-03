from __future__ import annotations

from typing import Any, Dict

import jax
import numpy as np


def _as_int(x: Any) -> int:
    """Convert JAX / NumPy scalar-like values to int."""
    try:
        return int(np.asarray(x))
    except (TypeError, ValueError, OverflowError):
        return int(x)


def _as_float(x: Any) -> float:
    """Convert JAX / NumPy scalar-like values to float."""
    try:
        return float(np.asarray(x))
    except (TypeError, ValueError, OverflowError):
        return float(x)


def extract_xtructure_signature(search_result: Any) -> Dict[str, Any]:
    """Extract xtr_* workload counters/ratios from a search result."""
    try:
        enabled = bool(jax.device_get(search_result.xtr_enabled))
    except (AttributeError, TypeError, ValueError):
        return {}

    if not enabled:
        return {}

    def get(name: str, default: int = 0) -> int:
        if not hasattr(search_result, name):
            return default
        return _as_int(jax.device_get(getattr(search_result, name)))

    steps = get("xtr_steps")
    cand_total = get("xtr_cand_total")
    cand_valid = get("xtr_cand_valid")
    cand_unique = get("xtr_cand_unique")
    accept = get("xtr_accept")
    ht_inserted = get("xtr_ht_inserted")
    ht_lookup = get("xtr_ht_lookup")
    ht_found = get("xtr_ht_found")
    pq_insert_calls = get("xtr_pq_insert_calls")
    pq_insert_items = get("xtr_pq_insert_items")
    pq_delete_calls = get("xtr_pq_delete_calls")
    pq_popped = get("xtr_pq_popped_items")
    pq_processed = get("xtr_pq_processed_items")
    pq_requeued = get("xtr_pq_requeued_items")
    pq_heap_sum = get("xtr_pq_heap_size_sum")
    pq_buffer_sum = get("xtr_pq_buffer_size_sum")
    pq_samples = get("xtr_pq_size_samples")

    dup_ratio = 0.0 if cand_valid == 0 else (1.0 - (cand_unique / cand_valid))
    hit_ratio = 0.0 if ht_lookup == 0 else (ht_found / ht_lookup)
    accept_ratio = 0.0 if cand_valid == 0 else (accept / cand_valid)
    pop_calls_eff = 0.0 if steps == 0 else (pq_delete_calls / steps)
    processed_ratio = 0.0 if pq_popped == 0 else (pq_processed / pq_popped)
    requeue_ratio = 0.0 if pq_popped == 0 else (pq_requeued / pq_popped)
    heap_avg = 0.0 if pq_samples == 0 else (pq_heap_sum / pq_samples)
    buffer_avg = 0.0 if pq_samples == 0 else (pq_buffer_sum / pq_samples)

    occupancy = None
    try:
        generated = _as_int(jax.device_get(search_result.generated_size))
        capacity = int(search_result.capacity)
        occupancy = float(generated) / float(capacity) if capacity > 0 else 0.0
    except (AttributeError, TypeError, ValueError):
        occupancy = None

    out: Dict[str, Any] = {
        "xtr_steps": steps,
        "xtr_cand_total": cand_total,
        "xtr_cand_valid": cand_valid,
        "xtr_cand_unique": cand_unique,
        "xtr_accept": accept,
        "xtr_ht_inserted": ht_inserted,
        "xtr_ht_lookup": ht_lookup,
        "xtr_ht_found": ht_found,
        "xtr_pq_insert_calls": pq_insert_calls,
        "xtr_pq_insert_items": pq_insert_items,
        "xtr_pq_delete_calls": pq_delete_calls,
        "xtr_pq_popped_items": pq_popped,
        "xtr_pq_processed_items": pq_processed,
        "xtr_pq_requeued_items": pq_requeued,
        "xtr_pq_heap_size_sum": pq_heap_sum,
        "xtr_pq_buffer_size_sum": pq_buffer_sum,
        "xtr_pq_size_samples": pq_samples,
        "xtr_dup_ratio_eff": _as_float(dup_ratio),
        "xtr_hit_ratio_eff": _as_float(hit_ratio),
        "xtr_accept_ratio": _as_float(accept_ratio),
        "xtr_pop_calls_eff": _as_float(pop_calls_eff),
        "xtr_processed_ratio_eff": _as_float(processed_ratio),
        "xtr_requeue_ratio_eff": _as_float(requeue_ratio),
        "xtr_pq_heap_size_avg": _as_float(heap_avg),
        "xtr_pq_buffer_size_avg": _as_float(buffer_avg),
    }
    if occupancy is not None:
        out["xtr_ht_occupancy_end"] = _as_float(occupancy)
    return out
