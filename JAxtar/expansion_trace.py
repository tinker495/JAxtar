"""Host-side expansion trace exposed by search result modules."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True, eq=False)
class ExpansionTrace:
    """Facts about the nodes a completed search expanded.

    Construction goes through ``from_raw``, which owns the expanded-node
    masking (``pop_generation > -1``), device-to-numpy conversion, and the
    zero-expanded -> ``None`` policy. The record deliberately carries no
    state payload; if a consumer ever needs states, the field is added here,
    not extracted in runners.
    """

    pop_generation: np.ndarray
    cost: np.ndarray
    dist: np.ndarray
    original_indices: np.ndarray
    parent_indices: np.ndarray | None = None
    solved_index: int | None = None

    @classmethod
    def from_raw(
        cls,
        *,
        pop_generation: Any,
        cost: Any,
        dist: Any,
        parent_indices: Any | None = None,
        solved_index: int | None = None,
    ) -> "ExpansionTrace | None":
        """Build an ExpansionTrace from full-capacity, hashtable-slot-indexed arrays."""
        pop = np.asarray(pop_generation)
        mask = pop > -1
        if not mask.any():
            return None
        return cls(
            pop_generation=pop[mask],
            cost=np.asarray(cost)[mask],
            dist=np.asarray(dist)[mask],
            original_indices=np.flatnonzero(mask),
            parent_indices=(None if parent_indices is None else np.asarray(parent_indices)[mask]),
            solved_index=solved_index,
        )
