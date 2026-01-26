"""
JAxtar Bidirectional Search Base Module
This module implements the core data structures and utilities for bidirectional A*/Q* search.

Bidirectional search explores from both start and goal simultaneously, reducing
the search space from O(b^d) to approximately O(b^(d/2)) where b is the branching
factor and d is the solution depth.
"""

from typing import Any

import chex
import jax.numpy as jnp
from puxle import Puzzle
from xtructure import FieldDescriptor, HashIdx, base_dataclass, xtructure_dataclass

from JAxtar.annotate import KEY_DTYPE
from JAxtar.stars.search_base import Current, SearchResult


@xtructure_dataclass
class MeetingPoint:
    """
    Tracks the best known meeting point between forward and backward search frontiers.

    Attributes:
        fwd_hashidx: Hash index of meeting state in forward hashtable
        bwd_hashidx: Hash index of meeting state in backward hashtable
        fwd_cost: g-value from start to meeting point (forward direction)
        bwd_cost: g-value from goal to meeting point (backward direction)
        total_cost: fwd_cost + bwd_cost (total path cost through this meeting point)
        found: Boolean indicating whether a valid meeting point has been discovered
    """

    fwd_hashidx: FieldDescriptor.scalar(dtype=HashIdx)
    bwd_hashidx: FieldDescriptor.scalar(dtype=HashIdx)
    fwd_cost: FieldDescriptor.scalar(dtype=KEY_DTYPE)
    bwd_cost: FieldDescriptor.scalar(dtype=KEY_DTYPE)
    total_cost: FieldDescriptor.scalar(dtype=KEY_DTYPE)
    found: FieldDescriptor.scalar(dtype=jnp.bool_)


@base_dataclass(static_fields=("action_size",))
class BiDirectionalSearchResult:
    """
    Container for bidirectional search state, holding both forward and backward
    SearchResult instances plus meeting point information.

    Attributes:
        forward: SearchResult for start -> goal direction
        backward: SearchResult for goal -> start direction
        action_size: Number of actions available (static)
        meeting: Best known meeting point between frontiers
    """

    forward: SearchResult
    backward: SearchResult
    action_size: int
    meeting: MeetingPoint

    @property
    def forward_capacity(self) -> int:
        return self.forward.capacity

    @property
    def backward_capacity(self) -> int:
        return self.backward.capacity

    @property
    def batch_size(self) -> int:
        return self.forward.batch_size

    @property
    def total_generated(self) -> int:
        """Total nodes generated across both directions."""
        return self.forward.generated_size + self.backward.generated_size


def build_bi_search_result(
    statecls: Puzzle.State,
    batch_size: int,
    max_nodes: int,
    action_size: int,
    pop_ratio: float = jnp.inf,
    min_pop: int = 1,
    parant_with_costs: bool = False,
) -> BiDirectionalSearchResult:
    """
    Creates a new BiDirectionalSearchResult with initialized forward and backward
    SearchResult instances.

    NOTE: This function should be called OUTSIDE of JIT-traced context,
    typically in the builder function before creating the JIT-compiled search.

    Args:
        statecls: The state class for the puzzle
        batch_size: Batch size for parallel processing
        max_nodes: Maximum nodes per direction (total will be 2x)
        action_size: Number of actions
        pop_ratio: Controls beam width
        min_pop: Minimum nodes to pop per batch
        parant_with_costs: Whether to use Parant_with_Costs for PQ values

    Returns:
        BiDirectionalSearchResult with initialized data structures
    """
    # Note: Both forward and backward use the default seed (42) in SearchResult.build.
    # This is acceptable since they are separate hash tables with independent state storage.
    # We pass pop_ratio and min_pop positionally to avoid triggering JIT tracing issues.
    forward = SearchResult.build(
        statecls,
        batch_size,
        max_nodes,
        action_size,
        pop_ratio,
        min_pop,
        parant_with_costs=parant_with_costs,
    )
    backward = SearchResult.build(
        statecls,
        batch_size,
        max_nodes,
        action_size,
        pop_ratio,
        min_pop,
        parant_with_costs=parant_with_costs,
    )

    # Initialize meeting point as not found with infinite cost
    meeting = MeetingPoint(
        fwd_hashidx=HashIdx.default(()),
        bwd_hashidx=HashIdx.default(()),
        fwd_cost=jnp.array(jnp.inf, dtype=KEY_DTYPE),
        bwd_cost=jnp.array(jnp.inf, dtype=KEY_DTYPE),
        total_cost=jnp.array(jnp.inf, dtype=KEY_DTYPE),
        found=jnp.array(False),
    )

    return BiDirectionalSearchResult(
        forward=forward,
        backward=backward,
        action_size=action_size,
        meeting=meeting,
    )


@base_dataclass
class BiLoopState:
    """
    Loop state for bidirectional search, carrying state for both directions.

    This is the main state object passed through the jax.lax.while_loop for
    bidirectional search algorithms.

    Attributes:
        bi_result: BiDirectionalSearchResult containing both search trees
        solve_config: Puzzle solve configuration
        params_forward: Heuristic/Q-function parameters for forward direction
        params_backward: Heuristic/Q-function parameters for backward direction
        current_forward: Current batch of nodes being processed (forward)
        current_backward: Current batch of nodes being processed (backward)
        filled_forward: Boolean mask for valid entries in forward batch
        filled_backward: Boolean mask for valid entries in backward batch
    """

    bi_result: BiDirectionalSearchResult
    solve_config: Puzzle.SolveConfig
    params_forward: Any
    params_backward: Any
    current_forward: Current
    current_backward: Current
    filled_forward: chex.Array
    filled_backward: chex.Array


@base_dataclass
class BiLoopStateWithStates:
    """
    Loop state for bidirectional search that also carries materialized states.

    Used for deferred variants where states are computed during pop and should
    be retained to avoid redundant hash table lookups.

    Attributes:
        bi_result: BiDirectionalSearchResult containing both search trees
        solve_config: Puzzle solve configuration
        params_forward: Heuristic/Q-function parameters for forward direction
        params_backward: Heuristic/Q-function parameters for backward direction
        current_forward: Current batch indices/costs (forward)
        current_backward: Current batch indices/costs (backward)
        states_forward: Materialized states for forward batch
        states_backward: Materialized states for backward batch
        filled_forward: Boolean mask for valid entries in forward batch
        filled_backward: Boolean mask for valid entries in backward batch
    """

    bi_result: BiDirectionalSearchResult
    solve_config: Puzzle.SolveConfig
    params_forward: Any
    params_backward: Any
    current_forward: Current
    current_backward: Current
    states_forward: Puzzle.State
    states_backward: Puzzle.State
    filled_forward: chex.Array
    filled_backward: chex.Array


def check_intersection(
    expanded_states: Puzzle.State,
    expanded_costs: chex.Array,
    expanded_mask: chex.Array,
    opposite_sr: SearchResult,
) -> tuple[chex.Array, HashIdx, chex.Array, chex.Array]:
    """
    Check if any expanded states exist in the opposite direction's hash table.

    This is the core operation for detecting when forward and backward frontiers
    meet. When a state is found in both hash tables, we have discovered a potential
    path from start to goal.

    Args:
        expanded_states: States that were just expanded
        expanded_costs: g-values for the expanded states (from this direction)
        expanded_mask: Boolean mask indicating valid expanded states
        opposite_sr: SearchResult for the opposite direction

    Returns:
        tuple of:
            - found_mask: Boolean mask where True indicates state exists in opposite HT
            - opposite_hashidx: Hash indices of found states in opposite HT
            - opposite_costs: g-values of found states in opposite direction
            - total_costs: Sum of costs from both directions (potential path cost)
    """
    # Look up expanded states in opposite hash table
    opposite_hashidx, found = opposite_sr.hashtable.lookup_parallel(expanded_states, expanded_mask)

    # Get costs from opposite direction for found states
    opposite_costs = opposite_sr.get_cost(opposite_hashidx)

    # Total path cost through this meeting point
    total_costs = expanded_costs + opposite_costs

    # Only consider valid intersections
    found_mask = jnp.logical_and(found, expanded_mask)

    return found_mask, opposite_hashidx, opposite_costs, total_costs


def update_meeting_point(
    meeting: MeetingPoint,
    found_mask: chex.Array,
    this_hashidxs: HashIdx,
    opposite_hashidxs: HashIdx,
    this_costs: chex.Array,
    opposite_costs: chex.Array,
    total_costs: chex.Array,
    is_forward: bool,
) -> MeetingPoint:
    """
    Update the meeting point if a better path is found.

    Args:
        meeting: Current best meeting point
        found_mask: Mask indicating which states intersect with opposite frontier
        this_hashidxs: Hash indices in the current direction's hash table
        opposite_hashidxs: Hash indices in the opposite direction's hash table
        this_costs: g-values from current direction
        opposite_costs: g-values from opposite direction
        total_costs: Sum of costs (this_costs + opposite_costs)
        is_forward: True if this direction is forward, False if backward

    Returns:
        Updated MeetingPoint with best known meeting point
    """
    import jax

    # Find the best meeting point among newly found intersections
    # Set non-found entries to inf so they don't affect argmin
    masked_total_costs = jnp.where(found_mask, total_costs, jnp.inf)
    best_new_idx = jnp.argmin(masked_total_costs)
    best_new_cost = masked_total_costs[best_new_idx]

    # Check if any valid intersection was found
    any_found = found_mask.any()

    # Update meeting point if new path is better
    better = jnp.logical_and(any_found, best_new_cost < meeting.total_cost)

    # Select appropriate indices based on direction
    if is_forward:
        new_fwd_hashidx = this_hashidxs[best_new_idx]
        new_bwd_hashidx = opposite_hashidxs[best_new_idx]
        new_fwd_cost = this_costs[best_new_idx]
        new_bwd_cost = opposite_costs[best_new_idx]
    else:
        new_fwd_hashidx = opposite_hashidxs[best_new_idx]
        new_bwd_hashidx = this_hashidxs[best_new_idx]
        new_fwd_cost = opposite_costs[best_new_idx]
        new_bwd_cost = this_costs[best_new_idx]

    # Use jax.lax.cond to conditionally update the meeting point
    # This handles HashIdx correctly since it returns one branch or the other
    def _update_meeting(_):
        return MeetingPoint(
            fwd_hashidx=new_fwd_hashidx,
            bwd_hashidx=new_bwd_hashidx,
            fwd_cost=new_fwd_cost,
            bwd_cost=new_bwd_cost,
            total_cost=best_new_cost,
            found=jnp.array(True),
        )

    def _keep_meeting(_):
        return MeetingPoint(
            fwd_hashidx=meeting.fwd_hashidx,
            bwd_hashidx=meeting.bwd_hashidx,
            fwd_cost=meeting.fwd_cost,
            bwd_cost=meeting.bwd_cost,
            total_cost=meeting.total_cost,
            found=jnp.logical_or(meeting.found, any_found),
        )

    return jax.lax.cond(better, _update_meeting, _keep_meeting, None)


def bi_termination_condition(
    bi_result: BiDirectionalSearchResult,
    fwd_min_f: chex.Array,
    bwd_min_f: chex.Array,
    cost_weight: float = 1.0,
) -> chex.Array:
    """
    Check the termination condition for bidirectional search.

    The search can terminate optimally when the weighted meeting point cost
    is <= both minimum f-values. This ensures:
    1. Any unexpanded forward state has f_fwd >= weighted_meeting_cost
    2. Any unexpanded backward state has f_bwd >= weighted_meeting_cost
    3. Therefore, no unexpanded path can have cost < meeting_cost

    Note: For optimal guarantees, cost_weight should be 1.0 or very close to it.
    With cost_weight < 1.0, the search is more greedy and may not find optimal.

    Args:
        bi_result: Current bidirectional search result
        fwd_min_f: Minimum f-value (cost_weight * g + h) in forward direction
        bwd_min_f: Minimum f-value (cost_weight * g + h) in backward direction
        cost_weight: Weight for path cost, must match PQ ordering

    Returns:
        Boolean indicating whether the search should terminate
    """
    # Best known path through meeting point (weighted for comparison with PQ keys)
    meeting_cost_weighted = cost_weight * bi_result.meeting.total_cost

    # Terminate when weighted meeting cost <= both min f-values
    # This proves no unexplored path can be better
    fwd_done = meeting_cost_weighted <= fwd_min_f
    bwd_done = meeting_cost_weighted <= bwd_min_f

    # Also need a meeting point to have been found
    optimal_found = jnp.logical_and(
        bi_result.meeting.found,
        jnp.logical_and(fwd_done, bwd_done),
    )

    return optimal_found


def get_min_f_value(
    sr: SearchResult,
    current: Current,
    filled: chex.Array,
    cost_weight: float = 1.0,
) -> chex.Array:
    """
    Get the minimum f-value from the current batch.

    For bidirectional search, we need to know the minimum f-value in each
    direction to check the termination condition. The f-value must match
    the PQ ordering: f = cost_weight * g + h.

    Args:
        sr: SearchResult containing cost and dist arrays
        current: Current batch of nodes
        filled: Boolean mask for valid entries
        cost_weight: Weight for path cost (must match PQ ordering)

    Returns:
        Minimum f-value among valid entries
    """
    # Get g-values and h-values (or Q-values) for current batch
    costs = current.cost  # g-values
    dists = sr.get_dist(current)  # h-values or Q-values

    # f = cost_weight * g + h (matches PQ ordering)
    f_values = cost_weight * costs + dists

    # Mask out invalid entries
    f_values = jnp.where(filled, f_values, jnp.inf)

    return jnp.min(f_values)
