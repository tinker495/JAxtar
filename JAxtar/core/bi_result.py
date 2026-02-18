"""
JAxtar Core Bidirectional Result Structures
"""


import chex
import jax
import jax.numpy as jnp
from puxle import Puzzle
from xtructure import FieldDescriptor, HashIdx, base_dataclass, xtructure_dataclass

from JAxtar.annotate import ACTION_DTYPE, HASH_SIZE_MULTIPLIER, KEY_DTYPE
from JAxtar.core.result import Current, Parent, SearchResult


@xtructure_dataclass
class MeetingPoint:
    """
    Tracks the best known meeting point between forward and backward search frontiers.
    """

    # Primary representation
    fwd_hashidx: FieldDescriptor.scalar(dtype=HashIdx)
    bwd_hashidx: FieldDescriptor.scalar(dtype=HashIdx)
    fwd_cost: FieldDescriptor.scalar(dtype=KEY_DTYPE)
    bwd_cost: FieldDescriptor.scalar(dtype=KEY_DTYPE)
    total_cost: FieldDescriptor.scalar(dtype=KEY_DTYPE)
    found: FieldDescriptor.scalar(dtype=jnp.bool_)

    # Extended representation for deferred variants:
    fwd_has_hashidx: FieldDescriptor.scalar(dtype=jnp.bool_)
    bwd_has_hashidx: FieldDescriptor.scalar(dtype=jnp.bool_)
    fwd_parent_hashidx: FieldDescriptor.scalar(dtype=HashIdx)
    fwd_parent_action: FieldDescriptor.scalar(dtype=ACTION_DTYPE)
    bwd_parent_hashidx: FieldDescriptor.scalar(dtype=HashIdx)
    bwd_parent_action: FieldDescriptor.scalar(dtype=ACTION_DTYPE)


@base_dataclass(static_fields=("action_size",))
class BiDirectionalSearchResult:
    """
    Container for bidirectional search state.
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
        return self.forward.generated_size + self.backward.generated_size

    @staticmethod
    def build(
        statecls: Puzzle.State,
        batch_size: int,
        max_nodes: int,
        action_size: int,
        pop_ratio: float = jnp.inf,
        min_pop: int = 1,
        pq_val_type: type = Current,
        hash_size_multiplier: int = HASH_SIZE_MULTIPLIER,
    ):
        forward = SearchResult.build(
            statecls,
            batch_size,
            max_nodes,
            action_size,
            pop_ratio,
            min_pop,
            pq_val_type=pq_val_type,
            hash_size_multiplier=hash_size_multiplier,
        )
        backward = SearchResult.build(
            statecls,
            batch_size,
            max_nodes,
            action_size,
            pop_ratio,
            min_pop,
            pq_val_type=pq_val_type,
            hash_size_multiplier=hash_size_multiplier,
        )

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
        return BiDirectionalSearchResult(forward, backward, action_size, meeting)


def check_intersection(
    expanded_states: Puzzle.State,
    expanded_costs: chex.Array,
    expanded_mask: chex.Array,
    opposite_sr: SearchResult,
) -> tuple[chex.Array, HashIdx, chex.Array, chex.Array]:
    """Check intersection with opposite direction."""
    opposite_hashidx, found = opposite_sr.hashtable.lookup_parallel(expanded_states, expanded_mask)
    opposite_costs = opposite_sr.get_cost(opposite_hashidx)
    total_costs = expanded_costs + opposite_costs
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
    """Update meeting point."""
    masked_total_costs = jnp.where(found_mask, total_costs, jnp.inf)
    best_new_idx = jnp.argmin(masked_total_costs)
    best_new_cost = masked_total_costs[best_new_idx]

    any_found = found_mask.any()
    better = jnp.logical_and(any_found, best_new_cost < meeting.total_cost)

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

    dummy_hashidx = HashIdx.default(())
    dummy_action = jnp.array(0, dtype=ACTION_DTYPE)

    def _update(_):
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

    def _keep(_):
        # We might need to carry over the 'found' status if previous was found
        # Actually meeting.found stores if we EVER found one.
        # But here we update specifically if better.
        # If not better, we keep old meeting point, but we might want to update found flag?
        # No, meeting stores "Best". If we found a worse one, we ignore it.
        return meeting

    return jax.lax.cond(better, _update, _keep, None)


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
    """Update meeting point for deferred variants without full insertion."""
    # Logic copied from bi_search_base.py

    this_best_costs = jnp.where(
        this_found,
        jnp.minimum(this_old_costs, candidate_costs),
        candidate_costs,
    ).astype(KEY_DTYPE)

    candidate_mask = jnp.logical_and(candidate_mask, this_best_costs < meeting.total_cost)

    opposite_hashidx, opposite_found = opposite_sr.hashtable.lookup_parallel(
        candidate_states, candidate_mask
    )
    opposite_costs = opposite_sr.get_cost(opposite_hashidx)
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
        best_this_cost = this_best_costs[best_idx]
        best_opposite_cost = opposite_costs[best_idx]

        dummy_hashidx = HashIdx.default(())
        dummy_action = jnp.array(0, dtype=ACTION_DTYPE)

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
                total_cost=best_total,
                found=jnp.array(True),
                fwd_has_hashidx=this_has_hashidx,
                bwd_has_hashidx=jnp.array(True),
                fwd_parent_hashidx=this_parent_hashidx_repr,
                fwd_parent_action=this_parent_action_repr,
                bwd_parent_hashidx=dummy_hashidx,
                bwd_parent_action=dummy_action,
            )
        else:
            return MeetingPoint(
                fwd_hashidx=best_opposite_hashidx,
                bwd_hashidx=this_hashidx_repr,
                fwd_cost=best_opposite_cost,
                bwd_cost=best_this_cost,
                total_cost=best_total,
                found=jnp.array(True),
                fwd_has_hashidx=jnp.array(True),
                bwd_has_hashidx=this_has_hashidx,
                fwd_parent_hashidx=dummy_hashidx,
                fwd_parent_action=dummy_action,
                bwd_parent_hashidx=this_parent_hashidx_repr,
                bwd_parent_action=this_parent_action_repr,
            )

    return jax.lax.cond(should_update, _do_update, lambda _: meeting, None)


def _add_batch_dim(state: Puzzle.State) -> Puzzle.State:
    return jax.tree_util.tree_map(lambda x: x[jnp.newaxis, ...], state)


def _strip_batch_dim(state: Puzzle.State) -> Puzzle.State:
    return jax.tree_util.tree_map(lambda x: x[0], state)


def materialize_meeting_point_hashidxs(
    bi_result: BiDirectionalSearchResult,
    puzzle: Puzzle,
    solve_config: Puzzle.SolveConfig,
) -> BiDirectionalSearchResult:
    """Materialize meeting point states if needed."""

    dummy_hashidx = HashIdx.default(())
    dummy_action = jnp.array(0, dtype=ACTION_DTYPE)

    def _compute_from_fwd_edge(bi_result, meeting):
        parent_state = bi_result.forward.hashtable[meeting.fwd_parent_hashidx]
        parent_b = _add_batch_dim(parent_state)
        action_b = jnp.array([meeting.fwd_parent_action], dtype=ACTION_DTYPE)
        filled_b = jnp.array([True])
        child_b, _ = puzzle.batched_get_actions(solve_config, parent_b, action_b, filled_b)
        return _strip_batch_dim(child_b)

    def _compute_from_bwd_edge(bi_result, meeting):
        parent_state = bi_result.backward.hashtable[meeting.bwd_parent_hashidx]
        parent_b = _add_batch_dim(parent_state)
        filled_b = jnp.array([True])
        inv_neigh, _ = puzzle.batched_get_inverse_neighbours(solve_config, parent_b, filled_b)
        a = meeting.bwd_parent_action.astype(jnp.int32)
        child = inv_neigh[a, 0]
        return child

    def _pick_meeting_state(bi_result, meeting):
        def _from_fwd(args):
            return args[0].forward.hashtable[args[1].fwd_hashidx]

        def _from_bwd(args):
            return args[0].backward.hashtable[args[1].bwd_hashidx]

        def _from_edge(args):
            return jax.lax.cond(
                args[1].fwd_has_hashidx,
                _from_fwd,
                lambda a: _compute_from_fwd_edge(a[0], a[1]),
                args,
            )

        args = (bi_result, meeting)
        return jax.lax.cond(
            meeting.fwd_has_hashidx,
            _from_fwd,
            lambda a: jax.lax.cond(a[1].bwd_has_hashidx, _from_bwd, _from_edge, a),
            args,
        )

    def _materialize_side(sr, meeting_state, parent_hashidx, parent_action, desired_cost):
        existing_hashidx, exists = sr.hashtable.lookup(meeting_state)

        def _use_existing(_):
            return sr, existing_hashidx

        def _insert_new(_):
            sr.hashtable, _, new_hashidx = sr.hashtable.insert(meeting_state)
            return sr, new_hashidx

        sr, hashidx = jax.lax.cond(exists, _use_existing, _insert_new, None)

        old_cost = sr.get_cost(hashidx)
        better = desired_cost < old_cost

        sr.cost = sr.cost.at[hashidx.index].set(
            jnp.where(better, desired_cost.astype(KEY_DTYPE), old_cost.astype(KEY_DTYPE))
        )
        sr.parent = sr.parent.at[hashidx.index].set_as_condition(
            better, Parent(hashidx=parent_hashidx, action=parent_action)
        )
        return sr, hashidx

    def _materialize_if_needed(bi_result):
        meeting = bi_result.meeting
        meeting_state = _pick_meeting_state(bi_result, meeting)

        def _mat_fwd(args):
            bi_res, m_state = args
            meeting0 = bi_res.meeting
            sr, hidx = _materialize_side(
                bi_res.forward,
                m_state,
                meeting0.fwd_parent_hashidx,
                meeting0.fwd_parent_action,
                meeting0.fwd_cost,
            )
            bi_res.forward = sr
            # Update meeting
            bi_res.meeting = meeting0.replace(
                fwd_hashidx=hidx,
                fwd_has_hashidx=jnp.array(True),
                fwd_parent_hashidx=dummy_hashidx,
                fwd_parent_action=dummy_action,
            )
            return bi_res

        def _mat_bwd(args):
            bi_res, m_state = args
            meeting0 = bi_res.meeting
            sr, hidx = _materialize_side(
                bi_res.backward,
                m_state,
                meeting0.bwd_parent_hashidx,
                meeting0.bwd_parent_action,
                meeting0.bwd_cost,
            )
            bi_res.backward = sr
            bi_res.meeting = meeting0.replace(
                bwd_hashidx=hidx,
                bwd_has_hashidx=jnp.array(True),
                bwd_parent_hashidx=dummy_hashidx,
                bwd_parent_action=dummy_action,
            )
            return bi_res

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
        # Refresh costs
        meeting = bi_result.meeting

        def _refresh(_):
            fwd_cost = bi_result.forward.get_cost(meeting.fwd_hashidx)
            bwd_cost = bi_result.backward.get_cost(meeting.bwd_hashidx)
            total = (fwd_cost + bwd_cost).astype(KEY_DTYPE)
            return meeting.replace(fwd_cost=fwd_cost, bwd_cost=bwd_cost, total_cost=total)

        bi_result.meeting = jax.lax.cond(meeting.found, _refresh, lambda _: meeting, None)
        return bi_result

    return jax.lax.cond(bi_result.meeting.found, _materialize_if_needed, lambda x: x, bi_result)


def finalize_bidirectional_result(
    bi_result: BiDirectionalSearchResult,
) -> BiDirectionalSearchResult:
    bi_result.forward.solved = bi_result.meeting.found
    bi_result.forward.solved_idx = Current(
        hashidx=bi_result.meeting.fwd_hashidx, cost=bi_result.meeting.fwd_cost
    )
    bi_result.backward.solved = bi_result.meeting.found
    bi_result.backward.solved_idx = Current(
        hashidx=bi_result.meeting.bwd_hashidx, cost=bi_result.meeting.bwd_cost
    )
    return bi_result


def bi_termination_condition(
    bi_result: BiDirectionalSearchResult,
    puzzle: Puzzle,
    solve_config: Puzzle.SolveConfig,
) -> chex.Array:
    """
    Check if bidirectional search should continue.
    Returns True if search should continue (not yet optimal/exhausted).
    """
    # 1. Check PQ emptiness
    fwd_empty = bi_result.forward.priority_queue.size == 0
    bwd_empty = bi_result.backward.priority_queue.size == 0
    search_exhausted = jnp.logical_and(fwd_empty, bwd_empty)

    # 2. Check Optimality
    # If meeting cost <= min_f_fwd + min_f_bwd, we found optimal.

    # Get min keys [1]
    # BGPQ structure: top_k_keys returns (k, batch).
    # We assume 'delete_mins' logic accesses min keys internally.
    # To implement this cleanly without breaking BGPQ abstraction:
    # Use helper method on SR or access PQ.

    # BGPQ usually has 'peek_mins'?
    # Assuming standard xtructure BGPQ:
    # `values, keys = pq.top_k(1)`?
    # Let's try `top_k_keys(1)`.

    # We need to handle empty PQ case (returns inf or similar?).

    bgpq_f = bi_result.forward.priority_queue
    bgpq_b = bi_result.backward.priority_queue

    k_f, _ = bgpq_f.top_k(1)  # keys, indices
    k_b, _ = bgpq_b.top_k(1)

    min_f = jnp.where(bgpq_f.size > 0, k_f[0], jnp.inf)
    min_b = jnp.where(bgpq_b.size > 0, k_b[0], jnp.inf)

    current_best = bi_result.meeting.total_cost

    # Condition: maintain search while (min_f + min_b) < current_best - epsilon
    # And search not exhausted.

    not_optimal = (min_f + min_b) < (current_best - 1e-6)

    should_continue = jnp.logical_and(jnp.logical_not(search_exhausted), not_optimal)

    return should_continue
