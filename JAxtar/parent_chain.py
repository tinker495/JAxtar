"""Host-side parent-pointer chain walking.

Single source of truth for walking a chain of parent records from a target
index back toward the root, with cycle/corruption protection via a max_steps
budget. Used by bidirectional reconstruction; future search algorithms that
need host-side parent walking should consume this helper rather than
re-deriving the loop.
"""

from __future__ import annotations

from typing import Any

import jax


INVALID_PARENT_SENTINEL: int = (1 << 32) - 1


def walk_parent_chain(
    parent_records: Any,
    start_index: int,
    max_steps: int,
    *,
    invalid_sentinel: int = INVALID_PARENT_SENTINEL,
) -> tuple[list[int], list[int]]:
    """Walk a parent-pointer chain from ``start_index`` toward the root.

    Args:
        parent_records: Any object supporting ``parent_records[idx]`` returning
            a Parent record with ``.hashidx.index`` and ``.action`` fields
            (xtructure-aware).
        start_index: Host-side index to begin walking from.
        max_steps: Upper bound on chain length. Exceeding it raises
            ``RuntimeError`` so a corrupt/cyclic parent table fails loudly
            rather than spinning.
        invalid_sentinel: Sentinel value that marks "no parent" (uint32 max by
            default — the xtructure convention for unset HashIdx).

    Returns:
        Tuple ``(indices, actions)`` in **target-to-root** order:
            indices[0] == start_index, indices[-1] is the root.
            actions[i] is the action that transitions ``indices[i + 1]`` to
            ``indices[i]``. ``len(actions) == len(indices) - 1`` when the
            walk reaches a root; equals ``len(indices)`` only if ``max_steps``
            is exceeded (which raises before returning).

    Callers that need root-to-target ordering should reverse both lists.
    """
    idx = int(start_index)
    indices: list[int] = [idx]
    actions: list[int] = []
    for _ in range(max_steps):
        parent = parent_records[idx]
        parent_idx = int(jax.device_get(parent.hashidx.index))
        if parent_idx == invalid_sentinel:
            return indices, actions
        actions.append(int(jax.device_get(parent.action)))
        idx = parent_idx
        indices.append(idx)
    raise RuntimeError("Path reconstruction exceeded max_steps (cycle/corruption suspected)")


__all__ = ["INVALID_PARENT_SENTINEL", "walk_parent_chain"]
