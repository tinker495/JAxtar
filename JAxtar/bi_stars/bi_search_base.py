"""
JAxtar Bidirectional Search Base Module
This module implements the core data structures and utilities for bidirectional A*/Q* search.

Bidirectional search explores from both start and goal simultaneously, reducing
the search space from O(b^d) to approximately O(b^(d/2)) where b is the branching
factor and d is the solution depth.

Shared hash table design
-------------------------
Both directions store their states in a single, shared `HashTable`. A given state
therefore occupies the *same* slot index regardless of which direction inserted it.
Two per-slot boolean arrays, `seen_forward` and `seen_backward`, record which
direction has registered (committed a g-value / parent for) each slot.

This turns meeting detection into a pure array gather: after a direction inserts a
batch it already knows each state's shared slot, so "did the opposite frontier reach
this state?" is just `seen_opposite[slot]`, and the opposite g-value is
`opposite.cost[slot]`. No second hash-table probe is needed.
"""

from typing import Any

import chex
import jax
import jax.numpy as jnp
import xtructure.numpy as xnp
from puxle import Puzzle
from xtructure import FieldDescriptor, HashIdx, base_dataclass, xtructure_dataclass

from JAxtar.annotate import ACTION_DTYPE, KEY_DTYPE
from JAxtar.parent_chain import walk_parent_chain
from JAxtar.solution_trace import (
    SolutionTrace,
    action_pad_int,
)
from JAxtar.stars.search_base import (
    Current,
    Parant_with_Costs,
    Parent,
    SearchResult,
    insert_priority_queue_batches,
)


@xtructure_dataclass
class MeetingPoint:
    """
    Tracks the best known meeting point between forward and backward search frontiers.

    With a shared hash table the meeting state lives at a single slot common to both
    directions, so `fwd_hashidx == bwd_hashidx`. Both fields are kept because path
    reconstruction walks the two directions' independent parent chains from that slot.

    Attributes:
        fwd_hashidx: Shared-table slot of the meeting state (forward view)
        bwd_hashidx: Shared-table slot of the meeting state (backward view); equals fwd_hashidx
        fwd_cost: g-value from start to meeting point (forward direction)
        bwd_cost: g-value from goal to meeting point (backward direction)
        total_cost: fwd_cost + bwd_cost (total path cost through this meeting point)
        found: Whether a valid meeting point has been discovered
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

    Both directions share a single hash table (`forward.hashtable is backward.hashtable`
    at loop boundaries); the `seen_*` arrays are indexed by that shared slot space and
    record per-direction registration.

    Attributes:
        forward: SearchResult for start -> goal direction
        backward: SearchResult for goal -> start direction
        action_size: Number of actions available (static)
        meeting: Best known meeting point between frontiers
        seen_forward: Per-slot mask of states registered by the forward direction
        seen_backward: Per-slot mask of states registered by the backward direction
    """

    forward: SearchResult
    backward: SearchResult
    action_size: int
    meeting: MeetingPoint
    seen_forward: chex.Array
    seen_backward: chex.Array

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
        """Distinct states generated across both directions.

        Forward and backward share one hash table, so its size already counts the
        union of states from both frontiers (`forward.generated_size` equals
        `backward.generated_size`).
        """
        return self.forward.generated_size

    def to_solution_trace(
        self,
        *,
        puzzle: Puzzle | None = None,
    ) -> SolutionTrace:
        """Return the host-side solution trace for CLI/evaluation adapters."""
        if not bool(jax.device_get(self.meeting.found)):
            return SolutionTrace.unsolved()

        action_pad = action_pad_int(ACTION_DTYPE)
        path = reconstruct_bidirectional_path(self, puzzle)
        if not path:
            return SolutionTrace.from_raw(
                solved=True,
                raw_actions=(),
                action_pad=action_pad,
            )

        return SolutionTrace.from_raw(
            solved=True,
            raw_actions=(action for action, _ in path[1:]),
            action_pad=action_pad,
            states=tuple(state for _, state in path),
            costs=None,
            dists=None,
        )


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
    SearchResult instances backed by a single shared hash table.

    NOTE: This helper is JAX-traceable and can be called inside a jitted
    `init_loop_state(...)` (mirroring `JAxtar.stars` patterns). Keep in mind it
    allocates full search buffers sized by `max_nodes`.

    Args:
        statecls: The state class for the puzzle
        batch_size: Batch size for parallel processing
        max_nodes: Shared hash-table capacity (both directions share this budget)
        action_size: Number of actions
        pop_ratio: Controls beam width
        min_pop: Minimum nodes to pop per batch
        parant_with_costs: Whether to use Parant_with_Costs for PQ values

    Returns:
        BiDirectionalSearchResult with initialized data structures
    """
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
    # Share ONE hash table between both directions so a state maps to a single slot
    # index across the whole search. Backward's own table (same seed/geometry) is
    # discarded; its cost/dist/parent buffers stay valid because they are sized by the
    # identical table geometry.
    backward.hashtable = forward.hashtable

    seen_shape = forward.hashtable.table.shape.batch
    seen_forward = jnp.zeros(seen_shape, dtype=jnp.bool_)
    seen_backward = jnp.zeros(seen_shape, dtype=jnp.bool_)

    dummy_hashidx = HashIdx.default(())
    meeting = MeetingPoint(
        fwd_hashidx=dummy_hashidx,
        bwd_hashidx=dummy_hashidx,
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
        seen_forward=seen_forward,
        seen_backward=seen_backward,
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


def _adopt_shared_hashtable(bi_result: BiDirectionalSearchResult, from_forward: bool):
    """Copy the shared hash table from one direction onto the other.

    Both directions must insert into the *same* table for slot indices to be shared.
    Since JAX structures are immutable, each expansion produces a new table object on
    the direction that ran; this baton-passes it to the other direction so the next
    insert (and `generated_size`) sees the up-to-date shared table.
    """
    if from_forward:
        bi_result.backward.hashtable = bi_result.forward.hashtable
    else:
        bi_result.forward.hashtable = bi_result.backward.hashtable
    return bi_result


def detect_meeting(
    bi_result: BiDirectionalSearchResult,
    hashidxs: HashIdx,
    mask: chex.Array,
    this_costs: chex.Array,
    is_forward: bool,
) -> tuple[chex.Array, chex.Array, chex.Array]:
    """Detect frontier meetings using the shared table + per-direction seen flags.

    `hashidxs` are shared-table slots for states this direction just registered. A
    state is a meeting point iff the opposite direction has also registered its slot,
    so the check and the opposite g-value are direct array gathers — no hash probe.

    Returns:
        found_mask: True where `mask` and the opposite frontier has registered the slot
        opposite_costs: opposite-direction g-values at those slots
        total_costs: this_costs + opposite_costs
    """
    slots = hashidxs.index
    if is_forward:
        seen_opposite = bi_result.seen_backward
        opposite_cost = bi_result.backward.cost
    else:
        seen_opposite = bi_result.seen_forward
        opposite_cost = bi_result.forward.cost

    found_mask = jnp.logical_and(mask, seen_opposite[slots])
    opposite_costs = opposite_cost[slots]
    total_costs = (this_costs + opposite_costs).astype(KEY_DTYPE)
    return found_mask, opposite_costs, total_costs


def register_seen(
    bi_result: BiDirectionalSearchResult,
    hashidxs: HashIdx,
    mask: chex.Array,
    is_forward: bool,
) -> BiDirectionalSearchResult:
    """Mark the given shared slots as registered in this direction (monotone OR)."""
    slots = hashidxs.index
    if is_forward:
        bi_result.seen_forward = bi_result.seen_forward.at[slots].max(mask)
    else:
        bi_result.seen_backward = bi_result.seen_backward.at[slots].max(mask)
    return bi_result


def update_meeting_point(
    meeting: MeetingPoint,
    found_mask: chex.Array,
    hashidxs: HashIdx,
    this_costs: chex.Array,
    opposite_costs: chex.Array,
    total_costs: chex.Array,
    is_forward: bool,
) -> MeetingPoint:
    """
    Update the meeting point if a better path is found.

    Args:
        meeting: Current best meeting point
        found_mask: Mask indicating which candidates intersect the opposite frontier
        hashidxs: Shared-table slots of the candidate states
        this_costs: g-values from the current direction
        opposite_costs: g-values from the opposite direction
        total_costs: this_costs + opposite_costs
        is_forward: True if this direction is forward, False if backward

    Returns:
        Updated MeetingPoint with the best known meeting point
    """
    # Pick the cheapest genuine intersection. Rank found candidates strictly below every
    # non-found slot (which sit at +inf), clipping so that a found candidate whose summed
    # cost overflowed KEY_DTYPE (float16, max ~65504) to +inf still outranks non-found —
    # otherwise argmin over all-+inf could land on a non-meeting slot.
    rank = jnp.where(found_mask, jnp.minimum(total_costs.astype(jnp.float32), 1e30), jnp.inf)
    best_new_idx = jnp.argmin(rank)
    best_new_cost = jnp.where(found_mask, total_costs, jnp.inf)[best_new_idx]

    any_found = found_mask.any()
    # Always record the FIRST genuine meeting, even if its summed cost overflowed float16
    # to +inf (initial meeting.total_cost is also +inf, so a strict `<` would reject it
    # and leave `found=True` pointing at the build-time sentinel slot). Afterwards, only
    # replace the recorded meeting when a strictly cheaper one appears.
    first_meeting = jnp.logical_and(any_found, jnp.logical_not(meeting.found))
    strictly_better = jnp.logical_and(any_found, best_new_cost < meeting.total_cost)
    record = jnp.logical_or(first_meeting, strictly_better)

    best_hashidx = hashidxs[best_new_idx]
    this_best = this_costs[best_new_idx].astype(KEY_DTYPE)
    opposite_best = opposite_costs[best_new_idx].astype(KEY_DTYPE)
    if is_forward:
        new_fwd_cost, new_bwd_cost = this_best, opposite_best
    else:
        new_fwd_cost, new_bwd_cost = opposite_best, this_best

    def _update_meeting(_):
        return MeetingPoint(
            fwd_hashidx=best_hashidx,
            bwd_hashidx=best_hashidx,
            fwd_cost=new_fwd_cost,
            bwd_cost=new_bwd_cost,
            total_cost=best_new_cost.astype(KEY_DTYPE),
            found=jnp.array(True),
        )

    def _keep_meeting(_):
        # `record` already covers the first-meeting case, so when we keep, `found` is
        # unchanged: either no meeting was found, or one was already recorded.
        return MeetingPoint(
            fwd_hashidx=meeting.fwd_hashidx,
            bwd_hashidx=meeting.bwd_hashidx,
            fwd_cost=meeting.fwd_cost,
            bwd_cost=meeting.bwd_cost,
            total_cost=meeting.total_cost,
            found=meeting.found,
        )

    return jax.lax.cond(record, _update_meeting, _keep_meeting, None)


def stamp_bi_solved_from_meeting(
    bi_result: BiDirectionalSearchResult,
) -> BiDirectionalSearchResult:
    """Stamp solved fields (`solved`, `solved_idx`) from `bi_result.meeting`.

    With a shared table the meeting always carries valid slots in both directions
    (both frontiers committed the state), so no materialization step is needed.
    """

    found = bi_result.meeting.found
    bi_result.forward.solved = found
    bi_result.forward.solved_idx = Current(
        hashidx=bi_result.meeting.fwd_hashidx,
        cost=bi_result.meeting.fwd_cost,
    )
    bi_result.backward.solved = found
    bi_result.backward.solved_idx = Current(
        hashidx=bi_result.meeting.bwd_hashidx,
        cost=bi_result.meeting.bwd_cost,
    )
    return bi_result


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

    # Check shared hash table capacity. Both directions share one table, so
    # `generated_size` is the shared total; when it is full neither can insert more.
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
    Initializes forward (from start) and backward (from goal) frontiers in the
    shared hash table and marks each root as registered in its direction.

    Returns:
       (fwd_filled, fwd_current, fwd_states, bwd_filled, bwd_current, bwd_states)
    """
    sr_batch_size = bi_result.batch_size

    # Forward root: insert start into the shared table.
    bi_result.forward.hashtable, _, fwd_hash_idx = bi_result.forward.hashtable.insert(start)
    bi_result.forward.cost = bi_result.forward.cost.at[fwd_hash_idx.index].set(0)

    # Backward root: adopt the shared table (now holding start), then insert goal.
    bi_result = _adopt_shared_hashtable(bi_result, from_forward=True)
    goal = puzzle.solve_config_to_state_transform(solve_config, key=jax.random.PRNGKey(0))
    bi_result.backward.hashtable, _, bwd_hash_idx = bi_result.backward.hashtable.insert(goal)
    bi_result.backward.cost = bi_result.backward.cost.at[bwd_hash_idx.index].set(0)

    # Re-sync forward to the shared table (now holding both roots).
    bi_result = _adopt_shared_hashtable(bi_result, from_forward=False)

    # Register roots in their respective directions.
    bi_result.seen_forward = bi_result.seen_forward.at[fwd_hash_idx.index].set(True)
    bi_result.seen_backward = bi_result.seen_backward.at[bwd_hash_idx.index].set(True)

    fwd_hash_idxs = xnp.pad(fwd_hash_idx, (0, sr_batch_size - 1))
    # Safety parity with `JAxtar.stars.search_base.init_base_loop_state_current`:
    # pad non-filled entries with +inf so an accidental read cannot produce work.
    fwd_costs = jnp.full((sr_batch_size,), jnp.inf, dtype=KEY_DTYPE).at[0].set(0)
    fwd_states = xnp.pad(start, (0, sr_batch_size - 1))
    fwd_filled = jnp.zeros(sr_batch_size, dtype=jnp.bool_).at[0].set(True)
    fwd_current = Current(hashidx=fwd_hash_idxs, cost=fwd_costs)

    bwd_hash_idxs = xnp.pad(bwd_hash_idx, (0, sr_batch_size - 1))
    bwd_costs = jnp.full((sr_batch_size,), jnp.inf, dtype=KEY_DTYPE).at[0].set(0)
    bwd_states = xnp.pad(goal, (0, sr_batch_size - 1))
    bwd_filled = jnp.zeros(sr_batch_size, dtype=jnp.bool_).at[0].set(True)
    bwd_current = Current(hashidx=bwd_hash_idxs, cost=bwd_costs)

    # start == goal ⟹ both roots land in the same shared slot.
    is_same = fwd_hash_idx.index == bwd_hash_idx.index

    bi_result.meeting = jax.lax.cond(
        is_same,
        lambda _: MeetingPoint(
            fwd_hashidx=fwd_hash_idx,
            bwd_hashidx=bwd_hash_idx,
            fwd_cost=jnp.array(0.0, dtype=KEY_DTYPE),
            bwd_cost=jnp.array(0.0, dtype=KEY_DTYPE),
            total_cost=jnp.array(0.0, dtype=KEY_DTYPE),
            found=jnp.array(True),
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

    def _walk(sr: SearchResult, target: HashIdx) -> tuple[list[int], list[int]]:
        max_steps = max(1, int(jax.device_get(sr.generated_size)) + 1)
        return walk_parent_chain(
            sr.parent,
            int(jax.device_get(target.index)),
            max_steps,
        )

    # Forward half: start -> meeting. walk_parent_chain returns target-to-root
    # order, so reverse to get root-to-target ordering for the merged path.
    fwd_t2r_indices, fwd_t2r_actions = _walk(bi_result.forward, bi_result.meeting.fwd_hashidx)
    fwd_indices = list(reversed(fwd_t2r_indices))
    fwd_actions = list(reversed(fwd_t2r_actions))
    fwd_states = [bi_result.forward.hashtable[HashIdx(index=jnp.uint32(i))] for i in fwd_indices]

    # Backward half: meeting -> goal (follow parent pointers toward the backward root).
    # Contract (puxle convention): the i-th inverse neighbour is a predecessor state from which
    # applying *forward* action i reaches the current state.
    # With that convention, the stored actions are already forward actions (no inversion needed).
    # If a puzzle violates this convention, reconstructed action sequences will be incorrect.
    bwd_indices, bwd_actions = _walk(bi_result.backward, bi_result.meeting.bwd_hashidx)
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


def build_bi_deferred_expand_direction(
    puzzle: Puzzle,
    cost_weight: float,
    look_ahead_pruning: bool,
    eval_fn: Any,
    *,
    is_forward: bool,
    use_heuristic_in_pop: bool,
):
    """
    Builds the generalized bidirectional deferred expansion logic.
    This acts as a high-order function replacing _expand_direction_deferred and _expand_direction_q.

    Deferred callback contract:
        Stars-parity contract:

        `eval_fn(puzzle, search_result, solve_config, params, states, costs, filled_tiles, filled,`
        `        look_ahead_pruning, cost_weight) -> (neighbour_keys, dists, optimal_mask)`

        - neighbour_keys: (action_size, batch), KEY_DTYPE
        - dists: (action_size, batch), KEY_DTYPE
        - optimal_mask: (action_size, batch), bool

        The bidirectional helper owns only orchestration:
        PQ insertion -> pop_full_with_actions -> post-pop meeting detection.
    """
    action_size = puzzle.action_size

    def _expand_direction(
        bi_result: BiDirectionalSearchResult,
        solve_config: Puzzle.SolveConfig,
        inverse_solveconfig: Puzzle.SolveConfig,
        params: Any,
        current: Current,
        states: Puzzle.State,
        filled: chex.Array,
    ) -> tuple[BiDirectionalSearchResult, Current, Puzzle.State, chex.Array]:
        if is_forward:
            search_result = bi_result.forward
            current_solve_config = solve_config
        else:
            search_result = bi_result.backward
            current_solve_config = inverse_solveconfig

        sr_batch_size = search_result.batch_size
        flat_size = action_size * sr_batch_size

        cost = current.cost
        hash_idx = current.hashidx

        idx_tiles = xnp.tile(hash_idx, (action_size, 1))
        action = jnp.tile(
            jnp.arange(action_size, dtype=ACTION_DTYPE)[:, jnp.newaxis],
            (1, sr_batch_size),
        )
        costs = jnp.tile(cost[jnp.newaxis, :], (action_size, 1))
        filled_tiles = jnp.tile(filled[jnp.newaxis, :], (action_size, 1))

        # Step 1: Evaluation Callback (algorithm-owned pruning/unique/optimal_mask)
        neighbour_keys, dists, optimal_mask = eval_fn(
            puzzle,
            search_result,
            current_solve_config,
            params,
            states,
            costs,
            filled_tiles,
            filled,
            look_ahead_pruning,
            cost_weight,
        )

        # Step 2: Priority Queue Insertion
        flattened_vals = Parant_with_Costs(
            parent=Parent(hashidx=idx_tiles.flatten(), action=action.flatten()),
            cost=costs.flatten(),
            dist=dists.flatten(),
        )
        flattened_keys = neighbour_keys.flatten()
        flattened_optimal_mask = optimal_mask.flatten()
        flattened_neighbour_keys = jnp.where(flattened_optimal_mask, flattened_keys, jnp.inf)

        sorted_key, sorted_idx = jax.lax.sort_key_val(
            flattened_neighbour_keys, jnp.arange(flat_size)
        )
        sorted_vals = flattened_vals[sorted_idx]
        sorted_optimal_mask = flattened_optimal_mask[sorted_idx]

        neighbour_keys_reshaped = sorted_key.reshape(action_size, sr_batch_size)
        vals_reshaped = sorted_vals.reshape((action_size, sr_batch_size))
        optimal_mask_reshaped = sorted_optimal_mask.reshape(action_size, sr_batch_size)

        search_result = insert_priority_queue_batches(
            search_result,
            neighbour_keys_reshaped,
            vals_reshaped,
            optimal_mask_reshaped,
        )

        # Step 3: Pop and commit into the shared hash table. The returned hashidxs are
        # shared-table slots, so meeting detection needs no extra hash probe.
        search_result, new_current, new_states, new_filled = search_result.pop_full_with_actions(
            puzzle=puzzle,
            solve_config=current_solve_config,
            use_heuristic=use_heuristic_in_pop,
            is_backward=not is_forward,
        )

        if is_forward:
            bi_result.forward = search_result
        else:
            bi_result.backward = search_result

        found_mask, opposite_costs, total_costs = detect_meeting(
            bi_result,
            new_current.hashidx,
            new_filled,
            new_current.cost,
            is_forward,
        )

        bi_result.meeting = update_meeting_point(
            bi_result.meeting,
            found_mask,
            new_current.hashidx,
            new_current.cost,
            opposite_costs,
            total_costs,
            is_forward,
        )

        bi_result = register_seen(bi_result, new_current.hashidx, new_filled, is_forward)

        return bi_result, new_current, new_states, new_filled

    return _expand_direction
