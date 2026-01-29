"""
JAxtar Bidirectional Search Base Module
This module implements the core data structures and utilities for bidirectional A*/Q* search.

Bidirectional search explores from both start and goal simultaneously, reducing
the search space from O(b^d) to approximately O(b^(d/2)) where b is the branching
factor and d is the solution depth.
"""

from typing import Any

import chex
import jax
import jax.numpy as jnp
import xtructure.numpy as xnp
from puxle import Puzzle
from xtructure import FieldDescriptor, HashIdx, base_dataclass, xtructure_dataclass

from JAxtar.annotate import ACTION_DTYPE, KEY_DTYPE
from JAxtar.stars.search_base import Current, Parent, SearchResult


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

    # Primary representation (legacy): both meeting states live in both hash tables.
    fwd_hashidx: FieldDescriptor.scalar(dtype=HashIdx)
    bwd_hashidx: FieldDescriptor.scalar(dtype=HashIdx)
    fwd_cost: FieldDescriptor.scalar(dtype=KEY_DTYPE)
    bwd_cost: FieldDescriptor.scalar(dtype=KEY_DTYPE)
    total_cost: FieldDescriptor.scalar(dtype=KEY_DTYPE)
    found: FieldDescriptor.scalar(dtype=jnp.bool_)

    # Extended representation for deferred variants:
    # During look-ahead we may discover a meeting state without inserting it into the
    # current direction's hash table. In that case we store the last edge (parent, action)
    # so the meeting state can be materialized later (or reconstructed without insertion).
    fwd_has_hashidx: FieldDescriptor.scalar(dtype=jnp.bool_)
    bwd_has_hashidx: FieldDescriptor.scalar(dtype=jnp.bool_)
    fwd_parent_hashidx: FieldDescriptor.scalar(dtype=HashIdx)
    fwd_parent_action: FieldDescriptor.scalar(dtype=ACTION_DTYPE)
    bwd_parent_hashidx: FieldDescriptor.scalar(dtype=HashIdx)
    bwd_parent_action: FieldDescriptor.scalar(dtype=ACTION_DTYPE)


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
    dummy_hashidx = HashIdx.default(())
    dummy_action = jnp.array(0, dtype=ACTION_DTYPE)
    meeting = MeetingPoint(
        fwd_hashidx=dummy_hashidx,
        bwd_hashidx=dummy_hashidx,
        fwd_cost=jnp.array(jnp.inf, dtype=KEY_DTYPE),
        bwd_cost=jnp.array(jnp.inf, dtype=KEY_DTYPE),
        total_cost=jnp.array(jnp.inf, dtype=KEY_DTYPE),
        found=jnp.array(False),
        fwd_has_hashidx=jnp.array(True),
        bwd_has_hashidx=jnp.array(True),
        fwd_parent_hashidx=dummy_hashidx,
        fwd_parent_action=dummy_action,
        bwd_parent_hashidx=dummy_hashidx,
        bwd_parent_action=dummy_action,
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
        solve_config: Puzzle solve configuration (for forward search)
        inverse_solveconfig: Puzzle solve configuration (for backward search)
        params_forward: Heuristic/Q-function parameters for forward direction
        params_backward: Heuristic/Q-function parameters for backward direction
        current_forward: Current batch of nodes being processed (forward)
        current_backward: Current batch of nodes being processed (backward)
        filled_forward: Boolean mask for valid entries in forward batch
        filled_backward: Boolean mask for valid entries in backward batch
    """

    bi_result: BiDirectionalSearchResult
    solve_config: Puzzle.SolveConfig
    inverse_solveconfig: Puzzle.SolveConfig
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
        solve_config: Puzzle solve configuration (for forward search)
        inverse_solveconfig: Puzzle solve configuration (for backward search)
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
    inverse_solveconfig: Puzzle.SolveConfig
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
    dummy_hashidx = HashIdx.default(())
    dummy_action = jnp.array(0, dtype=ACTION_DTYPE)

    def _update_meeting(_):
        return MeetingPoint(
            fwd_hashidx=new_fwd_hashidx,
            bwd_hashidx=new_bwd_hashidx,
            fwd_cost=new_fwd_cost,
            bwd_cost=new_bwd_cost,
            total_cost=best_new_cost,
            found=jnp.array(True),
            fwd_has_hashidx=jnp.array(True),
            bwd_has_hashidx=jnp.array(True),
            fwd_parent_hashidx=dummy_hashidx,
            fwd_parent_action=dummy_action,
            bwd_parent_hashidx=dummy_hashidx,
            bwd_parent_action=dummy_action,
        )

    def _keep_meeting(_):
        return MeetingPoint(
            fwd_hashidx=meeting.fwd_hashidx,
            bwd_hashidx=meeting.bwd_hashidx,
            fwd_cost=meeting.fwd_cost,
            bwd_cost=meeting.bwd_cost,
            total_cost=meeting.total_cost,
            found=jnp.logical_or(meeting.found, any_found),
            fwd_has_hashidx=meeting.fwd_has_hashidx,
            bwd_has_hashidx=meeting.bwd_has_hashidx,
            fwd_parent_hashidx=meeting.fwd_parent_hashidx,
            fwd_parent_action=meeting.fwd_parent_action,
            bwd_parent_hashidx=meeting.bwd_parent_hashidx,
            bwd_parent_action=meeting.bwd_parent_action,
        )

    return jax.lax.cond(better, _update_meeting, _keep_meeting, None)


def update_meeting_point_best_only_deferred(
    meeting: MeetingPoint,
    *,
    this_sr: SearchResult,
    opposite_sr: SearchResult,
    candidate_states: Puzzle.State,
    candidate_costs: chex.Array,
    candidate_mask: chex.Array,
    this_found: chex.Array,
    this_hashidx: HashIdx,
    this_old_costs: chex.Array,
    this_parent_hashidx: HashIdx,
    this_parent_action: chex.Array,
    is_forward: bool,
) -> MeetingPoint:
    """Best-only early meeting update for deferred variants.

    This updates the meeting point *without inserting* the meeting state into `this_sr`.
    If the meeting state already exists in `this_sr` (this_found=True), we store its hashidx.
    Otherwise we store the last edge (parent_hashidx, parent_action) so the state can be
    materialized later.
    """

    dummy_hashidx = HashIdx.default(())
    dummy_action = jnp.array(0, dtype=ACTION_DTYPE)

    opposite_hashidx, opposite_found = opposite_sr.hashtable.lookup_parallel(
        candidate_states, candidate_mask
    )
    opposite_costs = opposite_sr.get_cost(opposite_hashidx)

    this_best_costs = jnp.where(
        this_found,
        jnp.minimum(this_old_costs, candidate_costs),
        candidate_costs,
    ).astype(KEY_DTYPE)
    total_costs = (this_best_costs + opposite_costs).astype(KEY_DTYPE)

    meeting_candidate_mask = jnp.logical_and(opposite_found, candidate_mask)
    masked_total_costs = jnp.where(meeting_candidate_mask, total_costs, jnp.inf)
    best_idx = jnp.argmin(masked_total_costs)
    best_total = masked_total_costs[best_idx]

    should_update = jnp.logical_and(jnp.isfinite(best_total), best_total < meeting.total_cost)

    def _do_update(_):
        best_this_found = this_found[best_idx]

        best_this_hashidx = this_hashidx[best_idx]
        best_opposite_hashidx = opposite_hashidx[best_idx]

        best_parent_hashidx = this_parent_hashidx[best_idx]
        best_parent_action = this_parent_action[best_idx]

        best_this_cost = this_best_costs[best_idx].astype(KEY_DTYPE)
        best_opposite_cost = opposite_costs[best_idx].astype(KEY_DTYPE)
        best_total_cost = (best_this_cost + best_opposite_cost).astype(KEY_DTYPE)

        def _this_hash_repr(_):
            return best_this_hashidx, jnp.array(True), dummy_hashidx, dummy_action

        def _this_edge_repr(_):
            return dummy_hashidx, jnp.array(False), best_parent_hashidx, best_parent_action

        (
            this_hashidx_repr,
            this_has_hashidx,
            this_parent_hashidx_repr,
            this_parent_action_repr,
        ) = jax.lax.cond(best_this_found, _this_hash_repr, _this_edge_repr, None)

        if is_forward:
            return MeetingPoint(
                fwd_hashidx=this_hashidx_repr,
                bwd_hashidx=best_opposite_hashidx,
                fwd_cost=best_this_cost,
                bwd_cost=best_opposite_cost,
                total_cost=best_total_cost,
                found=jnp.array(True),
                fwd_has_hashidx=this_has_hashidx,
                bwd_has_hashidx=jnp.array(True),
                fwd_parent_hashidx=this_parent_hashidx_repr,
                fwd_parent_action=this_parent_action_repr,
                bwd_parent_hashidx=dummy_hashidx,
                bwd_parent_action=dummy_action,
            )

        return MeetingPoint(
            fwd_hashidx=best_opposite_hashidx,
            bwd_hashidx=this_hashidx_repr,
            fwd_cost=best_opposite_cost,
            bwd_cost=best_this_cost,
            total_cost=best_total_cost,
            found=jnp.array(True),
            fwd_has_hashidx=jnp.array(True),
            bwd_has_hashidx=this_has_hashidx,
            fwd_parent_hashidx=dummy_hashidx,
            fwd_parent_action=dummy_action,
            bwd_parent_hashidx=this_parent_hashidx_repr,
            bwd_parent_action=this_parent_action_repr,
        )

    return jax.lax.cond(should_update, _do_update, lambda _: meeting, None)


def materialize_meeting_point_hashidxs(
    bi_result: BiDirectionalSearchResult,
    puzzle: Puzzle,
    solve_config: Puzzle.SolveConfig,
) -> BiDirectionalSearchResult:
    """Ensure meeting point has hashidxs in both directions.

    For deferred variants we may have a meeting point represented via a last-edge
    (parent_hashidx, action) on one side. This function materializes the meeting
    state into the missing hash table(s) with at most one lookup+insert per side.
    """

    dummy_hashidx = HashIdx.default(())
    dummy_action = jnp.array(0, dtype=ACTION_DTYPE)

    def _add_batch_dim(state: Puzzle.State) -> Puzzle.State:
        return jax.tree_util.tree_map(lambda x: x[jnp.newaxis, ...], state)

    def _strip_batch_dim(state: Puzzle.State) -> Puzzle.State:
        return jax.tree_util.tree_map(lambda x: x[0], state)

    def _compute_from_fwd_edge(
        bi_result: BiDirectionalSearchResult, meeting: MeetingPoint
    ) -> Puzzle.State:
        parent_state = bi_result.forward.hashtable[meeting.fwd_parent_hashidx]
        parent_b = _add_batch_dim(parent_state)
        action_b = jnp.array([meeting.fwd_parent_action], dtype=ACTION_DTYPE)
        filled_b = jnp.array([True])
        child_b, _ = puzzle.batched_get_actions(solve_config, parent_b, action_b, filled_b)
        return _strip_batch_dim(child_b)

    def _compute_from_bwd_edge(
        bi_result: BiDirectionalSearchResult, meeting: MeetingPoint
    ) -> Puzzle.State:
        parent_state = bi_result.backward.hashtable[meeting.bwd_parent_hashidx]
        parent_b = _add_batch_dim(parent_state)
        filled_b = jnp.array([True])
        inv_neigh, _ = puzzle.batched_get_inverse_neighbours(solve_config, parent_b, filled_b)
        a = meeting.bwd_parent_action.astype(jnp.int32)
        child = inv_neigh[a, 0]
        return child

    def _pick_meeting_state(
        bi_result: BiDirectionalSearchResult, meeting: MeetingPoint
    ) -> Puzzle.State:
        # Prefer an existing hashidx (cheap); fallback to edge materialization.
        def _from_fwd(args):
            bi_result, meeting = args
            return bi_result.forward.hashtable[meeting.fwd_hashidx]

        def _from_bwd(args):
            bi_result, meeting = args
            return bi_result.backward.hashtable[meeting.bwd_hashidx]

        def _from_edge(args):
            bi_result, meeting = args
            return jax.lax.cond(
                meeting.fwd_has_hashidx,
                _from_fwd,
                lambda args: _compute_from_fwd_edge(args[0], args[1]),
                args,
            )

        args = (bi_result, meeting)
        return jax.lax.cond(
            meeting.fwd_has_hashidx,
            _from_fwd,
            lambda args: jax.lax.cond(meeting.bwd_has_hashidx, _from_bwd, _from_edge, args),
            args,
        )

    def _materialize_side(
        sr: SearchResult,
        meeting_state: Puzzle.State,
        parent_hashidx: HashIdx,
        parent_action: chex.Array,
        desired_cost: chex.Array,
    ) -> tuple[SearchResult, HashIdx]:
        existing_hashidx, exists = sr.hashtable.lookup(meeting_state)

        def _use_existing(_):
            return sr, existing_hashidx

        def _insert_new(_):
            sr.hashtable, _, new_hashidx = sr.hashtable.insert(meeting_state)
            return sr, new_hashidx

        sr, hashidx = jax.lax.cond(exists, _use_existing, _insert_new, None)

        # Ensure the stored g/parent are compatible with meeting reconstruction.
        old_cost = sr.get_cost(hashidx)
        better = desired_cost < old_cost

        sr.cost = sr.cost.at[hashidx.index].set(
            jnp.where(better, desired_cost.astype(KEY_DTYPE), old_cost.astype(KEY_DTYPE))
        )
        sr.parent = sr.parent.at[hashidx.index].set_as_condition(
            better,
            Parent(hashidx=parent_hashidx, action=parent_action),
        )
        return sr, hashidx

    def _materialize_if_needed(bi_result: BiDirectionalSearchResult) -> BiDirectionalSearchResult:
        meeting = bi_result.meeting
        meeting_state = _pick_meeting_state(bi_result, meeting)

        def _mat_fwd(args):
            bi_result, meeting_state = args
            meeting0 = bi_result.meeting
            sr, hidx = _materialize_side(
                bi_result.forward,
                meeting_state,
                meeting0.fwd_parent_hashidx,
                meeting0.fwd_parent_action,
                meeting0.fwd_cost,
            )
            bi_result.forward = sr
            bi_result.meeting = MeetingPoint(
                fwd_hashidx=hidx,
                bwd_hashidx=meeting0.bwd_hashidx,
                fwd_cost=meeting0.fwd_cost,
                bwd_cost=meeting0.bwd_cost,
                total_cost=meeting0.total_cost,
                found=meeting0.found,
                fwd_has_hashidx=jnp.array(True),
                bwd_has_hashidx=meeting0.bwd_has_hashidx,
                fwd_parent_hashidx=dummy_hashidx,
                fwd_parent_action=dummy_action,
                bwd_parent_hashidx=meeting0.bwd_parent_hashidx,
                bwd_parent_action=meeting0.bwd_parent_action,
            )
            return bi_result

        def _mat_bwd(args):
            bi_result, meeting_state = args
            meeting0 = bi_result.meeting
            sr, hidx = _materialize_side(
                bi_result.backward,
                meeting_state,
                meeting0.bwd_parent_hashidx,
                meeting0.bwd_parent_action,
                meeting0.bwd_cost,
            )
            bi_result.backward = sr
            bi_result.meeting = MeetingPoint(
                fwd_hashidx=meeting0.fwd_hashidx,
                bwd_hashidx=hidx,
                fwd_cost=meeting0.fwd_cost,
                bwd_cost=meeting0.bwd_cost,
                total_cost=meeting0.total_cost,
                found=meeting0.found,
                fwd_has_hashidx=meeting0.fwd_has_hashidx,
                bwd_has_hashidx=jnp.array(True),
                fwd_parent_hashidx=meeting0.fwd_parent_hashidx,
                fwd_parent_action=meeting0.fwd_parent_action,
                bwd_parent_hashidx=dummy_hashidx,
                bwd_parent_action=dummy_action,
            )
            return bi_result

        bi_result = jax.lax.cond(
            jnp.logical_and(meeting.found, jnp.logical_not(meeting.fwd_has_hashidx)),
            _mat_fwd,
            lambda x: x[0],
            (bi_result, meeting_state),
        )
        meeting = bi_result.meeting
        bi_result = jax.lax.cond(
            jnp.logical_and(meeting.found, jnp.logical_not(meeting.bwd_has_hashidx)),
            _mat_bwd,
            lambda x: x[0],
            (bi_result, meeting_state),
        )

        # Refresh costs from tables for consistency.
        meeting = bi_result.meeting

        def _refresh(_):
            fwd_cost = bi_result.forward.get_cost(meeting.fwd_hashidx)
            bwd_cost = bi_result.backward.get_cost(meeting.bwd_hashidx)
            total = (fwd_cost + bwd_cost).astype(KEY_DTYPE)
            return MeetingPoint(
                fwd_hashidx=meeting.fwd_hashidx,
                bwd_hashidx=meeting.bwd_hashidx,
                fwd_cost=fwd_cost.astype(KEY_DTYPE),
                bwd_cost=bwd_cost.astype(KEY_DTYPE),
                total_cost=total,
                found=meeting.found,
                fwd_has_hashidx=jnp.array(True),
                bwd_has_hashidx=jnp.array(True),
                fwd_parent_hashidx=dummy_hashidx,
                fwd_parent_action=dummy_action,
                bwd_parent_hashidx=dummy_hashidx,
                bwd_parent_action=dummy_action,
            )

        bi_result.meeting = jax.lax.cond(
            meeting.found,
            _refresh,
            lambda _: meeting,
            None,
        )
        return bi_result

    return jax.lax.cond(bi_result.meeting.found, _materialize_if_needed, lambda x: x, bi_result)


def bi_termination_condition(
    bi_result: BiDirectionalSearchResult,
    fwd_min_f: chex.Array,
    bwd_min_f: chex.Array,
    cost_weight: float = 1.0,
    terminate_on_first_solution: bool = True,
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
        terminate_on_first_solution: If True, terminate as soon as any path is found

    Returns:
        Boolean indicating whether the search should terminate
    """
    if terminate_on_first_solution:
        return bi_result.meeting.found

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
    """Get the minimum f-value from the current batch.

    For bidirectional search, we need a per-direction lower bound consistent with
    the priority queue ordering. This function assumes the PQ key is of the form:

        f = cost_weight * g + dist

    where:
    - `g` is `Current.cost` (the path cost stored in the hashtable for the popped state).
    - `dist` is `sr.get_dist(current)`.

    Important contract: `dist` must be in the same units as `g` and must be the
    *same quantity* that was used to order the PQ for this algorithm.
    - A*/A* variants: dist is typically the heuristic h(state).
    - Q* deferred variants: `SearchResult` stores a heuristic-like value for popped
      states (commonly h(state) = Q(parent, action) - step_cost(parent->state)).

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


def common_bi_loop_condition(
    bi_result: BiDirectionalSearchResult,
    filled_forward: chex.Array,
    filled_backward: chex.Array,
    current_forward: Current,
    current_backward: Current,
    cost_weight: float,
    terminate_on_first_solution: bool,
) -> chex.Array:
    """
    Common loop condition for bidirectional search.

    Continues while:
    1. At least one direction can still expand nodes (has frontier AND hashtable capacity)
    2. Termination condition not met (lower_bound < upper_bound)
    """
    # Check if queues have nodes
    fwd_has_nodes = filled_forward.any()
    bwd_has_nodes = filled_backward.any()

    # Check hash table capacity per direction.
    # If one direction is full, we can still expand the other direction and
    # potentially intersect with the already-built frontier.
    fwd_not_full = bi_result.forward.generated_size < bi_result.forward.capacity
    bwd_not_full = bi_result.backward.generated_size < bi_result.backward.capacity
    has_work = jnp.logical_or(
        jnp.logical_and(fwd_has_nodes, fwd_not_full),
        jnp.logical_and(bwd_has_nodes, bwd_not_full),
    )

    # Check termination condition
    fwd_min_f = get_min_f_value(bi_result.forward, current_forward, filled_forward, cost_weight)
    bwd_min_f = get_min_f_value(bi_result.backward, current_backward, filled_backward, cost_weight)

    should_terminate = bi_termination_condition(
        bi_result, fwd_min_f, bwd_min_f, cost_weight, terminate_on_first_solution
    )

    return jnp.logical_and(has_work, ~should_terminate)


def initialize_bi_loop_common(
    bi_result: BiDirectionalSearchResult,
    puzzle: Puzzle,
    solve_config: Puzzle.SolveConfig,
    start: Puzzle.State,
) -> tuple[chex.Array, Current, Puzzle.State, chex.Array, Current, Puzzle.State]:
    """
    Common initialization logic for bidirectional search loop state.
    Initializes forward (from start) and backward (from goal) frontiers.

    Returns:
       (fwd_filled, fwd_current, fwd_states, bwd_filled, bwd_current, bwd_states)
    """
    sr_batch_size = bi_result.batch_size

    # Initialize forward search (from start)
    bi_result.forward.hashtable, _, fwd_hash_idx = bi_result.forward.hashtable.insert(start)
    bi_result.forward.cost = bi_result.forward.cost.at[fwd_hash_idx.index].set(0)

    fwd_hash_idxs = xnp.pad(fwd_hash_idx, (0, sr_batch_size - 1))
    fwd_costs = jnp.zeros((sr_batch_size,), dtype=KEY_DTYPE)
    fwd_states = xnp.pad(start, (0, sr_batch_size - 1))
    fwd_filled = jnp.zeros(sr_batch_size, dtype=jnp.bool_).at[0].set(True)
    fwd_current = Current(hashidx=fwd_hash_idxs, cost=fwd_costs)

    # Initialize backward search (from goal)
    # Use puzzle-level transform to obtain a concrete goal state.
    goal = puzzle.solve_config_to_state_transform(solve_config, key=jax.random.PRNGKey(0))
    bi_result.backward.hashtable, _, bwd_hash_idx = bi_result.backward.hashtable.insert(goal)
    bi_result.backward.cost = bi_result.backward.cost.at[bwd_hash_idx.index].set(0)

    bwd_hash_idxs = xnp.pad(bwd_hash_idx, (0, sr_batch_size - 1))
    bwd_costs = jnp.zeros((sr_batch_size,), dtype=KEY_DTYPE)
    bwd_states = xnp.pad(goal, (0, sr_batch_size - 1))
    bwd_filled = jnp.zeros(sr_batch_size, dtype=jnp.bool_).at[0].set(True)
    bwd_current = Current(hashidx=bwd_hash_idxs, cost=bwd_costs)

    # Check if start == goal (cost = 0 case)
    start_in_bwd_idx, start_in_bwd_found = bi_result.backward.hashtable.lookup(start)
    is_same = jnp.logical_and(start_in_bwd_found, start_in_bwd_idx.index == bwd_hash_idx.index)

    dummy_hashidx = fwd_hash_idx
    dummy_action = jnp.array(0, dtype=ACTION_DTYPE)

    bi_result.meeting = jax.lax.cond(
        is_same,
        lambda _: MeetingPoint(
            fwd_hashidx=fwd_hash_idx,
            bwd_hashidx=bwd_hash_idx,
            fwd_cost=jnp.array(0.0, dtype=KEY_DTYPE),
            bwd_cost=jnp.array(0.0, dtype=KEY_DTYPE),
            total_cost=jnp.array(0.0, dtype=KEY_DTYPE),
            found=jnp.array(True),
            fwd_has_hashidx=jnp.array(True),
            bwd_has_hashidx=jnp.array(True),
            fwd_parent_hashidx=dummy_hashidx,
            fwd_parent_action=dummy_action,
            bwd_parent_hashidx=dummy_hashidx,
            bwd_parent_action=dummy_action,
        ),
        lambda _: bi_result.meeting,
        None,
    )

    return fwd_filled, fwd_current, fwd_states, bwd_filled, bwd_current, bwd_states


def reconstruct_bidirectional_path(
    bi_result: BiDirectionalSearchResult,
    puzzle: Puzzle,
) -> list[tuple[int, Puzzle.State]]:
    """
    Reconstruct the full path from start to goal using the meeting point.

    The return value is a sequence of (action, state) pairs along the solution.
    The first element corresponds to the start state and uses action = -1.
    For i >= 1, `action` is the forward action taken to reach `state` from the
    previous state.

    Args:
        bi_result: BiDirectionalSearchResult from bidirectional search
        puzzle: Puzzle instance

    Returns:
        List of (action, state) pairs from start to goal.
    """
    if not bi_result.meeting.found:
        return []

    def _u32_max() -> int:
        # Sentinel used by xtructure for "-1" index (uint32 max).
        return (1 << 32) - 1

    def _trace_root_to_target(sr: SearchResult, target: HashIdx) -> tuple[list[int], list[int]]:
        """Return (indices, actions) where indices are root->target inclusive."""
        idx = int(jax.device_get(target.index))
        max_steps = max(1, int(jax.device_get(sr.generated_size)) + 1)
        indices_rev: list[int] = [idx]
        actions_rev: list[int] = []
        for _ in range(max_steps):
            parent = sr.parent[idx]
            parent_idx = int(jax.device_get(parent.hashidx.index))
            if parent_idx == _u32_max():
                break
            actions_rev.append(int(jax.device_get(parent.action)))
            idx = parent_idx
            indices_rev.append(idx)
        else:
            raise RuntimeError(
                "Path reconstruction exceeded max_steps (cycle/corruption suspected)"
            )
        return list(reversed(indices_rev)), list(reversed(actions_rev))

    def _trace_target_to_root(sr: SearchResult, start_idx: HashIdx) -> tuple[list[int], list[int]]:
        """Return (indices, actions) where indices are start->root inclusive."""
        idx = int(jax.device_get(start_idx.index))
        max_steps = max(1, int(jax.device_get(sr.generated_size)) + 1)
        indices: list[int] = [idx]
        actions: list[int] = []
        for _ in range(max_steps):
            parent = sr.parent[idx]
            parent_idx = int(jax.device_get(parent.hashidx.index))
            if parent_idx == _u32_max():
                break
            actions.append(int(jax.device_get(parent.action)))
            idx = parent_idx
            indices.append(idx)
        else:
            raise RuntimeError(
                "Path reconstruction exceeded max_steps (cycle/corruption suspected)"
            )
        return indices, actions

    # Forward half: start -> meeting
    fwd_indices, fwd_actions = _trace_root_to_target(
        bi_result.forward, bi_result.meeting.fwd_hashidx
    )
    fwd_states = [bi_result.forward.hashtable[HashIdx(index=jnp.uint32(i))] for i in fwd_indices]

    # Backward half: meeting -> goal (follow parent pointers toward the backward root)
    # Contract (puxle convention): the i-th inverse neighbour is a predecessor state from which
    # applying *forward* action i reaches the current state.
    # With that convention, the stored actions are already forward actions (no inversion needed).
    # If a puzzle violates this convention, reconstructed action sequences will be incorrect.
    bwd_indices, bwd_actions = _trace_target_to_root(
        bi_result.backward, bi_result.meeting.bwd_hashidx
    )
    bwd_states = [bi_result.backward.hashtable[HashIdx(index=jnp.uint32(i))] for i in bwd_indices]

    # Merge, dropping the duplicated meeting state in the backward half.
    states = fwd_states + bwd_states[1:]
    actions = fwd_actions + bwd_actions

    if len(states) == 0:
        return []

    path: list[tuple[int, Puzzle.State]] = [(-1, states[0])]
    for a, s in zip(actions, states[1:]):
        path.append((int(a), s))
    return path
