"""
JAxtar Core Loop Builder
"""

from typing import Any

import chex
import jax.numpy as jnp
import xtructure.numpy as xnp
from puxle import Puzzle
from xtructure import base_dataclass

from JAxtar.annotate import KEY_DTYPE
from JAxtar.core.common import finalize_search_result
from JAxtar.core.result import Current, SearchResult
from JAxtar.core.search_strategy import ExpansionPolicy


@base_dataclass
class LoopState:
    search_result: SearchResult
    solve_config: Puzzle.SolveConfig
    heuristic_params: Any
    current: Any  # Current or ParentWithCosts (popped item)
    filled: chex.Array  # Mask for valid items in current batch
    states: Puzzle.State  # Materialized states for current (avoid re-materialization)


def unified_search_loop_builder(
    puzzle: Puzzle,
    expansion_policy: ExpansionPolicy,
    batch_size: int,
    max_nodes: int,
    pop_ratio: float,
    min_pop: int,
    pq_val_type: type = Current,
):
    """
    Builds the initialization, condition, and body for a unified search loop.
    """
    action_size = puzzle.action_size
    statecls = puzzle.State

    def init_loop_state(
        solve_config: Puzzle.SolveConfig,
        start: Puzzle.State,
        heuristic_params: Any = None,
        **kwargs,
    ) -> LoopState:
        # Build Search Result
        search_result = SearchResult.build(
            statecls,
            batch_size,
            max_nodes,
            action_size,
            pop_ratio,
            min_pop,
            pq_val_type=pq_val_type,
        )

        # 1. Insert Start into HT
        (
            search_result.hashtable,
            _,
            hash_idx,
        ) = search_result.hashtable.insert(start)

        # 2. Set Cost
        search_result.cost = search_result.cost.at[hash_idx.index].set(0)

        sr_batch_size = search_result.batch_size

        # Start Batch
        start_hash_idxs = xnp.pad(hash_idx, (0, sr_batch_size - 1))
        start_costs = jnp.full((sr_batch_size,), jnp.inf, dtype=KEY_DTYPE).at[0].set(0)

        start_filled = jnp.zeros(sr_batch_size, dtype=jnp.bool_).at[0].set(True)
        start_current = Current(hashidx=start_hash_idxs, cost=start_costs)

        # 3. Materialize start states (padding)
        start_states = xnp.pad(start, (0, sr_batch_size - 1))

        return LoopState(
            search_result=search_result,
            solve_config=solve_config,
            heuristic_params=heuristic_params,
            current=start_current,
            filled=start_filled,
            states=start_states,
        )

    def loop_condition(loop_state: LoopState):
        search_result = loop_state.search_result
        has_frontier = jnp.logical_or(
            jnp.any(loop_state.filled),
            search_result.priority_queue.size > 0,
        )
        has_capacity = search_result.generated_size < search_result.capacity
        has_work = jnp.logical_and(has_frontier, has_capacity)

        solved_now = jnp.any(
            jnp.logical_and(
                puzzle.batched_is_solved(loop_state.solve_config, loop_state.states),
                loop_state.filled,
            )
        )
        should_continue = jnp.logical_and(
            jnp.logical_not(search_result.solved),
            jnp.logical_not(solved_now),
        )
        return jnp.logical_and(has_work, should_continue)

    def loop_body(loop_state: LoopState) -> LoopState:
        new_sr, new_current, new_states, new_filled = expansion_policy.expand(
            loop_state.search_result,
            puzzle,
            loop_state.solve_config,
            loop_state.heuristic_params,
            loop_state.current,
            loop_state.filled,
        )

        # Check if any newly popped states are solutions and update solved flag
        solved_mask = jnp.logical_and(
            puzzle.batched_is_solved(loop_state.solve_config, new_states), new_filled
        )
        new_sr = finalize_search_result(new_sr, new_current, solved_mask)

        return LoopState(
            search_result=new_sr,
            solve_config=loop_state.solve_config,
            heuristic_params=loop_state.heuristic_params,
            current=new_current,
            filled=new_filled,
            states=new_states,
        )

    return init_loop_state, loop_condition, loop_body
