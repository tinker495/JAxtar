from typing import Any

import jax
import jax.numpy as jnp
import xtructure.numpy as xnp
from puxle import Puzzle

from helpers.jax_compile import jit_with_warmup
from heuristic.heuristic_base import Heuristic
from JAxtar.annotate import ACTION_DTYPE, KEY_DTYPE, MIN_BATCH_SIZE
from JAxtar.beamsearch.search_base import (
    ACTION_PAD,
    BeamSearchLoopState,
    BeamSearchResult,
    apply_selected_candidates,
    beam_loop_continue_if_not_solved,
    finalize_beam_search_result,
    non_backtracking_mask,
    select_beam,
)
from JAxtar.utils.array_ops import stable_partition_three
from JAxtar.utils.batch_switcher import variable_batch_switcher_builder


def _heuristic_beam_loop_builder(
    puzzle: Puzzle,
    heuristic: Heuristic,
    batch_size: int = 1024,
    max_nodes: int = int(1e6),
    pop_ratio: float = jnp.inf,
    cost_weight: float = 1.0 - 1e-6,
    non_backtracking_steps: int = 3,
):

    statecls = puzzle.State
    action_size = puzzle.action_size
    beam_width = batch_size
    denom = max(1, action_size // 2)
    min_keep = max(1, beam_width // denom)
    pop_ratio = float(pop_ratio)
    max_depth = max(1, (max_nodes + beam_width - 1) // beam_width)
    variable_heuristic_batch_switcher = variable_batch_switcher_builder(
        heuristic.batched_distance,
        max_batch_size=beam_width,
        min_batch_size=MIN_BATCH_SIZE,
        pad_value=jnp.inf,
    )

    if non_backtracking_steps < 0:
        raise ValueError("non_backtracking_steps must be non-negative")
    non_backtracking_steps = int(non_backtracking_steps)

    def init_loop_state(solve_config: Puzzle.SolveConfig, start: Puzzle.State, **kwargs):
        result = BeamSearchResult.build(
            statecls,
            beam_width,
            max_depth,
            action_size,
        )
        heuristic_parameters = heuristic.prepare_heuristic_parameters(solve_config, **kwargs)

        # Seed the beam with the start state
        result.beam = result.beam.at[0].set(start)
        result.cost = result.cost.at[0].set(0)
        init_filled = jnp.zeros(beam_width, dtype=jnp.bool_).at[0].set(True)
        start_dist = variable_heuristic_batch_switcher(
            heuristic_parameters, result.beam, init_filled
        ).astype(KEY_DTYPE)[0]
        start_score = (cost_weight * result.cost[0] + start_dist).astype(KEY_DTYPE)
        result.dist = result.dist.at[0].set(start_dist)
        result.scores = result.scores.at[0].set(start_score)
        result.parent_index = result.parent_index.at[0].set(-1)
        result.active_trace = result.active_trace.at[0].set(0)
        result.trace_cost = result.trace_cost.at[0].set(0)
        result.trace_dist = result.trace_dist.at[0].set(start_dist)
        result.trace_depth = result.trace_depth.at[0].set(0)
        result.trace_action = result.trace_action.at[0].set(ACTION_PAD)
        result.trace_state = result.trace_state.at[0].set(start)
        return BeamSearchLoopState(
            search_result=result,
            solve_config=solve_config,
            params=heuristic_parameters,
        )

    def loop_condition(loop_state: BeamSearchLoopState):
        return beam_loop_continue_if_not_solved(
            loop_state.search_result,
            puzzle,
            loop_state.solve_config,
        )

    def loop_body(loop_state: BeamSearchLoopState):
        search_result = loop_state.search_result
        solve_config = loop_state.solve_config
        heuristic_parameters = loop_state.params
        filled_mask = search_result.filled_mask()
        beam_states = search_result.beam
        sr_beam_width = search_result.beam_width
        sr_action_size = search_result.action_size
        child_shape = (sr_action_size, sr_beam_width)
        flat_count = sr_action_size * sr_beam_width

        neighbours, transition_cost = puzzle.batched_get_neighbours(
            solve_config, beam_states, filled_mask
        )

        base_costs = search_result.cost
        child_costs = (base_costs[jnp.newaxis, :] + transition_cost).astype(KEY_DTYPE)
        child_valid = jnp.logical_and(filled_mask[jnp.newaxis, :], jnp.isfinite(child_costs))
        child_costs = jnp.where(child_valid, child_costs, jnp.inf)

        flat_states = neighbours.reshape((flat_count,))
        flat_cost = child_costs.reshape((flat_count,))
        flat_valid = child_valid.reshape((flat_count,))

        unique_flat_mask = xnp.unique_mask(
            flat_states,
            key=flat_cost,
            filled=flat_valid,
        )
        unique_mask = unique_flat_mask.reshape(child_shape)
        child_valid = jnp.logical_and(child_valid, unique_mask)

        if non_backtracking_steps:
            parent_trace_matrix = jnp.broadcast_to(
                search_result.active_trace[jnp.newaxis, :], child_shape
            )
            flat_parent_trace = parent_trace_matrix.reshape(flat_count)
            allowed_mask = non_backtracking_mask(
                flat_states,
                flat_parent_trace,
                search_result.trace_state,
                search_result.trace_parent,
                non_backtracking_steps,
            ).reshape(child_shape)
            child_valid = jnp.logical_and(child_valid, allowed_mask)

        child_costs = jnp.where(child_valid, child_costs, jnp.inf)

        flat_states_tree = neighbours.reshape((flat_count,))
        flat_valid = child_valid.reshape(flat_count)
        global_perm = stable_partition_three(
            flat_valid, jnp.zeros_like(flat_valid, dtype=jnp.bool_)
        )
        ordered_states_tree = xnp.take(flat_states_tree, global_perm, axis=0)
        ordered_valid = jnp.take(flat_valid, global_perm, axis=0)

        num_chunks = sr_action_size
        chunk_states_tree = ordered_states_tree.reshape((num_chunks, sr_beam_width))
        chunk_valid = ordered_valid.reshape((num_chunks, sr_beam_width))

        chunk_dists = jnp.full((num_chunks, sr_beam_width), jnp.inf, dtype=KEY_DTYPE)

        def _compute_chunk(i, acc):
            row_mask = chunk_valid[i]

            def _calc(_):
                chunk_states = chunk_states_tree[i]
                dist_row = variable_heuristic_batch_switcher(
                    heuristic_parameters,
                    chunk_states,
                    row_mask,
                ).astype(KEY_DTYPE)
                return jnp.where(row_mask, dist_row, jnp.inf)

            dist_row = jax.lax.cond(
                jnp.any(row_mask),
                _calc,
                lambda _: acc[i],
                None,
            )
            return acc.at[i].set(dist_row)

        dists_compacted = jax.lax.fori_loop(0, num_chunks, _compute_chunk, chunk_dists)
        ordered_dists = dists_compacted.reshape((flat_count,))
        flat_dists = jnp.full(flat_count, jnp.inf, dtype=KEY_DTYPE)
        flat_dists = flat_dists.at[global_perm].set(ordered_dists)
        dists = flat_dists.reshape(child_shape)

        scores = (cost_weight * child_costs + dists).astype(KEY_DTYPE)
        scores = jnp.where(child_valid, scores, jnp.inf)

        flat_states = neighbours.reshape((flat_count,))
        flat_cost = child_costs.reshape((flat_count,))
        flat_dist = dists.reshape((flat_count,))
        flat_scores = scores.reshape((flat_count,))
        flat_valid = child_valid.reshape((flat_count,))

        selected_scores, selected_idx, keep_mask = select_beam(
            flat_scores,
            sr_beam_width,
            pop_ratio=pop_ratio,
            min_keep=min_keep,
        )

        selected_states = flat_states[selected_idx]
        selected_costs = flat_cost[selected_idx]
        selected_dists = flat_dist[selected_idx]
        selected_actions = (selected_idx // sr_beam_width).astype(ACTION_DTYPE)
        selected_parents = (selected_idx % sr_beam_width).astype(jnp.int32)
        selected_valid = jnp.logical_and(keep_mask, flat_valid[selected_idx])
        unique_valid = xnp.unique_mask(
            selected_states,
            key=selected_scores,
            filled=selected_valid,
        )
        selected_valid = jnp.logical_and(selected_valid, unique_valid)

        search_result = apply_selected_candidates(
            search_result,
            selected_states,
            selected_costs,
            selected_dists,
            selected_scores,
            selected_actions,
            selected_parents,
            selected_valid,
        )
        return BeamSearchLoopState(
            search_result=search_result,
            solve_config=solve_config,
            params=heuristic_parameters,
        )

    return init_loop_state, loop_condition, loop_body


def beam_builder(
    puzzle: Puzzle,
    heuristic: Heuristic,
    batch_size: int = 1024,
    max_nodes: int = int(1e6),
    pop_ratio: float = jnp.inf,
    cost_weight: float = 1.0 - 1e-6,
    show_compile_time: bool = False,
    non_backtracking_steps: int = 3,
    warmup_inputs: tuple[Puzzle.SolveConfig, Puzzle.State] | None = None,
):
    """Construct a batched heuristic beam-search solver."""

    init_loop_state, loop_condition, loop_body = _heuristic_beam_loop_builder(
        puzzle,
        heuristic,
        batch_size,
        max_nodes,
        pop_ratio,
        cost_weight,
        non_backtracking_steps,
    )

    def beam(
        solve_config: Puzzle.SolveConfig,
        start: Puzzle.State,
        **kwargs: Any,
    ) -> BeamSearchResult:
        loop_state = init_loop_state(solve_config, start, **kwargs)
        loop_state = jax.lax.while_loop(loop_condition, loop_body, loop_state)
        return finalize_beam_search_result(
            loop_state.search_result,
            puzzle,
            solve_config,
        )

    return jit_with_warmup(
        beam,
        puzzle=puzzle,
        show_compile_time=show_compile_time,
        warmup_inputs=warmup_inputs,
    )
