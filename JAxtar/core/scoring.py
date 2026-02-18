"""
JAxtar Core Scoring Policies
"""

import chex
from xtructure import base_dataclass

from JAxtar.annotate import KEY_DTYPE
from JAxtar.core.search_strategy import ScoringPolicy


@base_dataclass
class AStarScoring(ScoringPolicy):
    """
    Standard A* Scoring: f = cost_weight * g + h
    """

    def compute_priority(
        self,
        cost: chex.Array,
        dist: chex.Array,
        cost_weight: float,
    ) -> chex.Array:
        # Avoid inf * 0 or similar if possible, but JAX handles it.
        # cost_weight * cost + dist
        return (cost_weight * cost + dist).astype(KEY_DTYPE)


@base_dataclass
class QStarScoring(ScoringPolicy):
    """
    Q* Scoring: f = cost_weight * g + (Q or h)
    Can support pessimistic updates logic if needed, but here we just compute f.
    Update logic is usually in expansion.
    """

    def compute_priority(
        self,
        cost: chex.Array,
        q_value: chex.Array,
        cost_weight: float,
    ) -> chex.Array:
        return (cost_weight * cost + q_value).astype(KEY_DTYPE)
