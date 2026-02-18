"""
JAxtar Core Bidirectional Loop Builder
"""

from typing import Any

import chex
import jax
import jax.numpy as jnp
import xtructure.numpy as xnp
from puxle import Puzzle
from xtructure import base_dataclass

from JAxtar.annotate import KEY_DTYPE
from JAxtar.core.bi_result import (
    BiDirectionalSearchResult,
    materialize_meeting_point_hashidxs,
)
from JAxtar.core.result import Current
from JAxtar.core.search_strategy import ExpansionPolicy


@base_dataclass
class BiLoopState:
    bi_result: BiDirectionalSearchResult
    solve_config: Puzzle.SolveConfig
    inverse_solve_config: (Puzzle.SolveConfig)  # Need this for backward search (e.g. goal as start)
    heuristic_params: Any
    inverse_heuristic_params: Any
    current_fwd: Any  # Current or ParentWithCosts
    current_bwd: Any
    filled_fwd: chex.Array
    filled_bwd: chex.Array


def unified_bi_search_loop_builder(
    puzzle: Puzzle,
    fwd_expansion_policy: ExpansionPolicy,
    bwd_expansion_policy: ExpansionPolicy,
    batch_size: int,
    max_nodes: int,
    pop_ratio: float,
    min_pop: int,
    pq_val_type: type = Current,
    terminate_on_first_solution: bool = True,
):
    """
    Builds initialization, condition, and body for unified bidirectional search.
    """
    action_size = puzzle.action_size
    statecls = puzzle.State

    def init_loop_state(
        solve_config: Puzzle.SolveConfig,
        inverse_solve_config: Puzzle.SolveConfig,
        start: Puzzle.State,
        goal: Puzzle.State,  # Goal state for backward search start
        heuristic_params: Any = None,
        inverse_heuristic_params: Any = None,
        **kwargs,
    ) -> BiLoopState:

        bi_result = BiDirectionalSearchResult.build(
            statecls,
            batch_size,
            max_nodes,
            action_size,
            pop_ratio,
            min_pop,
            pq_val_type=pq_val_type,
        )

        # 1. Insert Start to Forward
        (
            bi_result.forward.hashtable,
            _,
            fwd_hash_idx,
        ) = bi_result.forward.hashtable.insert(start)
        bi_result.forward.cost = bi_result.forward.cost.at[fwd_hash_idx.index].set(0)

        # 2. Insert Goal to Backward
        (
            bi_result.backward.hashtable,
            _,
            bwd_hash_idx,
        ) = bi_result.backward.hashtable.insert(goal)
        bi_result.backward.cost = bi_result.backward.cost.at[bwd_hash_idx.index].set(0)

        # 3. Initial Batches
        sr_batch_size = bi_result.batch_size

        # Forward
        fwd_idxs = xnp.pad(fwd_hash_idx, (0, sr_batch_size - 1))
        fwd_costs = jnp.zeros((sr_batch_size,), dtype=KEY_DTYPE)
        fwd_filled = jnp.zeros(sr_batch_size, dtype=jnp.bool_).at[0].set(True)
        current_fwd = Current(hashidx=fwd_idxs, cost=fwd_costs)

        # Backward
        bwd_idxs = xnp.pad(bwd_hash_idx, (0, sr_batch_size - 1))
        bwd_costs = jnp.zeros((sr_batch_size,), dtype=KEY_DTYPE)
        bwd_filled = jnp.zeros(sr_batch_size, dtype=jnp.bool_).at[0].set(True)
        current_bwd = Current(hashidx=bwd_idxs, cost=bwd_costs)

        # Trivial solved case: start is already goal.
        # Initialize meeting immediately so terminate_on_first_solution exits with zero cost.
        start_batched = jax.tree_util.tree_map(lambda x: x[jnp.newaxis, ...], start)
        start_is_goal = jnp.any(puzzle.batched_is_solved(solve_config, start_batched))

        def _set_trivial_meeting(meeting):
            meeting.fwd_hashidx = fwd_hash_idx
            meeting.bwd_hashidx = bwd_hash_idx
            meeting.fwd_cost = jnp.array(0, dtype=KEY_DTYPE)
            meeting.bwd_cost = jnp.array(0, dtype=KEY_DTYPE)
            meeting.total_cost = jnp.array(0, dtype=KEY_DTYPE)
            meeting.found = jnp.array(True)
            meeting.fwd_has_hashidx = jnp.array(True)
            meeting.bwd_has_hashidx = jnp.array(True)
            return meeting

        bi_result.meeting = jax.lax.cond(
            start_is_goal,
            _set_trivial_meeting,
            lambda meeting: meeting,
            bi_result.meeting,
        )

        return BiLoopState(
            bi_result=bi_result,
            solve_config=solve_config,
            inverse_solve_config=inverse_solve_config,
            heuristic_params=heuristic_params,
            inverse_heuristic_params=inverse_heuristic_params,
            current_fwd=current_fwd,
            current_bwd=current_bwd,
            filled_fwd=fwd_filled,
            filled_bwd=bwd_filled,
        )

    def loop_condition(loop_state: BiLoopState):
        bi_result = loop_state.bi_result
        fwd_has_frontier = jnp.logical_or(
            jnp.any(loop_state.filled_fwd),
            bi_result.forward.priority_queue.size > 0,
        )
        bwd_has_frontier = jnp.logical_or(
            jnp.any(loop_state.filled_bwd),
            bi_result.backward.priority_queue.size > 0,
        )
        fwd_not_full = bi_result.forward.generated_size < bi_result.forward.capacity
        bwd_not_full = bi_result.backward.generated_size < bi_result.backward.capacity
        has_work = jnp.logical_or(
            jnp.logical_and(fwd_has_frontier, fwd_not_full),
            jnp.logical_and(bwd_has_frontier, bwd_not_full),
        )

        if terminate_on_first_solution:
            should_terminate = bi_result.meeting.found
        else:
            fwd_weight = getattr(fwd_expansion_policy, "cost_weight", 1.0)
            bwd_weight = getattr(bwd_expansion_policy, "cost_weight", 1.0)

            def _get_min_f(sr, cur, filled, weight):
                if hasattr(cur, "dist"):
                    dist = cur.dist
                else:
                    dist = sr.get_dist(cur)
                f = weight * cur.cost + dist
                return jnp.min(jnp.where(filled, f, jnp.inf))

            fwd_min_f = _get_min_f(
                bi_result.forward,
                loop_state.current_fwd,
                loop_state.filled_fwd,
                fwd_weight,
            )
            bwd_min_f = _get_min_f(
                bi_result.backward,
                loop_state.current_bwd,
                loop_state.filled_bwd,
                bwd_weight,
            )

            meeting_cost_weighted = fwd_weight * bi_result.meeting.total_cost
            fwd_done = meeting_cost_weighted <= fwd_min_f
            bwd_done = meeting_cost_weighted <= bwd_min_f
            should_terminate = jnp.logical_and(
                bi_result.meeting.found,
                jnp.logical_and(fwd_done, bwd_done),
            )

        return jnp.logical_and(has_work, jnp.logical_not(should_terminate))

    def loop_body(loop_state: BiLoopState) -> BiLoopState:
        bi_result = loop_state.bi_result
        fwd_config = loop_state.solve_config
        bwd_config = loop_state.inverse_solve_config

        # 1. Forward Expansion
        # Pass meeting point
        (
            bi_result.forward,
            bi_result.meeting,
            next_fwd,
            next_fwd_states,  # Unused if not needed for intersection (expand_bi handled it)
            next_fwd_filled,
        ) = fwd_expansion_policy.expand_bi(
            bi_result.forward,
            bi_result.backward,  # Opponent
            bi_result.meeting,
            puzzle,
            fwd_config,
            loop_state.heuristic_params,
            loop_state.current_fwd,
            loop_state.filled_fwd,
            is_forward=True,
        )

        # 2. Backward Expansion
        (
            bi_result.backward,
            bi_result.meeting,
            next_bwd,
            next_bwd_states,
            next_bwd_filled,
        ) = bwd_expansion_policy.expand_bi(
            bi_result.backward,
            bi_result.forward,  # Opponent
            bi_result.meeting,
            puzzle,
            bwd_config,  # Inverse config
            loop_state.inverse_heuristic_params,
            loop_state.current_bwd,
            loop_state.filled_bwd,
            is_forward=False,
        )

        # 3. Materialize Meeting Point (if needed, e.g. deferred edge)
        # expand_bi might update meeting to "Edge" format.
        # We need to ensure we can reconstruct path.
        # materialize_meeting_point_hashidxs does this ensuring HT entries exist.
        bi_result = materialize_meeting_point_hashidxs(bi_result, puzzle, fwd_config)

        return BiLoopState(
            bi_result=bi_result,
            solve_config=fwd_config,
            inverse_solve_config=bwd_config,
            heuristic_params=loop_state.heuristic_params,
            inverse_heuristic_params=loop_state.inverse_heuristic_params,
            current_fwd=next_fwd,
            current_bwd=next_bwd,
            filled_fwd=next_fwd_filled,
            filled_bwd=next_bwd_filled,
        )

    return init_loop_state, loop_condition, loop_body
