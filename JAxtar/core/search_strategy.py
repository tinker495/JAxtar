"""
JAxtar Core Search Strategy Protocols
"""

from typing import Any, Protocol, runtime_checkable

import chex
from puxle import Puzzle

from JAxtar.core.result import Current, SearchResult

# Forward references for type hints (avoid circular imports if files were split differently)
# But since we use Protocol, it's fine.
# Note: MeetingPoint is in JAxtar.core.bi_result.
# We should import it only for type hint if we can, or use string.


@runtime_checkable
class ExpansionPolicy(Protocol):
    """
    Protocol for expanding nodes in the search.
    Handles the generation of children/actions and their insertion into
    Wait lists (Priority Queue) or Closed lists (Hash Table).
    """

    def expand(
        self,
        search_result: SearchResult,
        puzzle: Puzzle,
        solve_config: Puzzle.SolveConfig,
        heuristic_params: Any,
        current: Current,
        filled: chex.Array,
        **kwargs,
    ) -> tuple[SearchResult, Current, Puzzle.State, chex.Array]:
        """
        Expand the current batch of nodes.

        Args:
           search_result: Current search state
           puzzle: Puzzle definition
           solve_config: Configuration for solving (e.g. goal)
           heuristic_params: Parameters for the heuristic function
           current: Current batch of nodes to expand (popped from PQ)
           filled: Mask indicating valid nodes in the batch

        Returns:
            Tuple containing:
            - Updated SearchResult
            - Next batch of nodes to process (popped from PQ in the next step)
            - Materialized states for the next batch
            - Mask for the next batch
        """
        ...

    def expand_bi(
        self,
        search_result: SearchResult,
        opposite_search_result: SearchResult,
        meeting_point: Any,  # "MeetingPoint"
        puzzle: Puzzle,
        solve_config: Puzzle.SolveConfig,
        heuristic_params: Any,
        current: Current,
        filled: chex.Array,
        is_forward: bool,
        **kwargs,
    ) -> tuple[
        SearchResult, Any, Current, Puzzle.State, chex.Array
    ]:  # SR, Meeting, NextCurrent, NextStates, NextFilled
        """
        Performs expansion for bidirectional search, handling termination checks.
        Default implementation delegates to expand and assumes external intersection check.
        Can be overridden for optimized Lookahead intersection checks.
        """
        ...


@runtime_checkable
class ScoringPolicy(Protocol):
    """
    Protocol for calculating priority scores (f-values) and keys for the PQ.
    """

    def compute_priority(
        self,
        cost: chex.Array,
        heuristic_value: chex.Array,
        cost_weight: float,
    ) -> chex.Array:
        """
        Compute the priority key for the priority queue.
        Typically f = weight * g + h.
        """
        ...
