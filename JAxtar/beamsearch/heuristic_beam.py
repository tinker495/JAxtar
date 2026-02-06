import time
from typing import Any

import jax
import jax.numpy as jnp
import xtructure.numpy as xnp
from puxle import Puzzle

from helpers.jax_compile import compile_with_example
from heuristic.heuristic_base import Heuristic
from JAxtar.annotate import ACTION_DTYPE, KEY_DTYPE, MIN_BATCH_UNIT
from JAxtar.beamsearch.search_base import (
    ACTION_PAD,
    TRACE_INDEX_DTYPE,
    TRACE_INVALID,
    BeamSearchLoopState,
    BeamSearchResult,
    non_backtracking_mask,
    select_beam,
)
from JAxtar.utils.batch_switcher import (
    build_batch_sizes_for_cap,
    variable_batch_switcher_builder,
)


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
    heuristic_batch_sizes = build_batch_sizes_for_cap(beam_width, min_batch_unit=MIN_BATCH_UNIT)

    variable_heuristic_batch_switcher = variable_batch_switcher_builder(
        heuristic.batched_distance,
        pad_value=jnp.inf,
        batch_sizes=heuristic_batch_sizes,
        partition_mode="auto",
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
        search_result = loop_state.search_result
        solve_config = loop_state.solve_config
        filled_mask = search_result.filled_mask()
        has_states = filled_mask.any()
        depth_ok = search_result.depth < search_result.max_depth

        beam_states = search_result.beam
        solved = puzzle.batched_is_solved(solve_config, beam_states)
        solved = jnp.logical_and(solved, filled_mask)
        return jnp.logical_and(jnp.logical_and(depth_ok, has_states), ~solved.any())

    def loop_body(loop_state: BeamSearchLoopState):
        search_result = loop_state.search_result
        solve_config = loop_state.solve_config
        heuristic_parameters = loop_state.params
        filled_mask = search_result.filled_mask()
        beam_states = search_result.beam
        sr_beam_width = search_result.beam_width
        sr_action_size = search_result.action_size
        sr_max_depth = search_result.max_depth
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

        parent_matrix = jnp.broadcast_to(
            jnp.arange(sr_beam_width, dtype=jnp.int32), (sr_action_size, sr_beam_width)
        )

        # Keep (action, batch) layout so row-partition skip can avoid empty action rows.
        dists = variable_heuristic_batch_switcher(
            heuristic_parameters,
            neighbours,
            child_valid,
        ).astype(KEY_DTYPE)
        dists = jnp.where(child_valid, dists, jnp.inf)

        scores = (cost_weight * child_costs + dists).astype(KEY_DTYPE)
        scores = jnp.where(child_valid, scores, jnp.inf)

        flat_states = neighbours.reshape((flat_count,))
        flat_cost = child_costs.reshape((flat_count,))
        flat_dist = dists.reshape((flat_count,))
        flat_scores = scores.reshape((flat_count,))
        flat_valid = child_valid.reshape((flat_count,))
        flat_parents = parent_matrix.reshape((flat_count,))

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
        selected_parents = flat_parents[selected_idx]
        selected_valid = jnp.logical_and(keep_mask, flat_valid[selected_idx])
        unique_valid = xnp.unique_mask(
            selected_states,
            key=selected_scores,
            filled=selected_valid,
        )
        selected_valid = jnp.logical_and(selected_valid, unique_valid)

        parent_trace_ids = search_result.active_trace[selected_parents]

        selected_costs = jnp.where(selected_valid, selected_costs, jnp.inf)
        selected_dists = jnp.where(selected_valid, selected_dists, jnp.inf)
        selected_scores = jnp.where(selected_valid, selected_scores, jnp.inf)
        selected_actions = selected_actions.astype(ACTION_DTYPE)

        invalid_parent = jnp.full_like(parent_trace_ids, TRACE_INVALID)
        parent_trace_ids = jnp.where(selected_valid, parent_trace_ids, invalid_parent)

        next_depth_idx = jnp.minimum(search_result.depth + 1, sr_max_depth)
        trace_offset = next_depth_idx.astype(TRACE_INDEX_DTYPE) * jnp.asarray(
            sr_beam_width, dtype=TRACE_INDEX_DTYPE
        )
        slot_indices = jnp.arange(sr_beam_width, dtype=TRACE_INDEX_DTYPE)
        next_trace_ids = trace_offset + slot_indices

        trace_actions = jnp.where(
            selected_valid,
            selected_actions,
            jnp.full_like(selected_actions, ACTION_PAD),
        )
        depth_fill = jnp.full((sr_beam_width,), next_depth_idx, dtype=jnp.int32)
        depth_default = -jnp.ones((sr_beam_width,), dtype=jnp.int32)
        trace_depths = jnp.where(selected_valid, depth_fill, depth_default)

        search_result.trace_parent = search_result.trace_parent.at[next_trace_ids].set(
            parent_trace_ids
        )
        search_result.trace_action = search_result.trace_action.at[next_trace_ids].set(
            trace_actions
        )
        search_result.trace_cost = search_result.trace_cost.at[next_trace_ids].set(selected_costs)
        search_result.trace_dist = search_result.trace_dist.at[next_trace_ids].set(selected_dists)
        search_result.trace_depth = search_result.trace_depth.at[next_trace_ids].set(trace_depths)
        search_result.trace_state = search_result.trace_state.at[next_trace_ids].set(
            selected_states
        )

        invalid_trace = jnp.full_like(next_trace_ids, TRACE_INVALID)
        next_active_trace = jnp.where(selected_valid, next_trace_ids, invalid_trace)

        search_result.beam = selected_states
        search_result.cost = selected_costs
        search_result.dist = selected_dists
        search_result.scores = selected_scores
        search_result.parent_index = jnp.where(
            selected_valid, selected_parents, -jnp.ones_like(selected_parents)
        )
        search_result.active_trace = next_active_trace
        selected_count = selected_valid.astype(jnp.int32).sum()
        search_result.generated_size = search_result.generated_size + selected_count
        search_result.depth = search_result.depth + 1

        # Keep the while_loop carry structure consistent with the input
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
        puzzle, heuristic, batch_size, max_nodes, pop_ratio, cost_weight, non_backtracking_steps
    )

    def beam(
        solve_config: Puzzle.SolveConfig,
        start: Puzzle.State,
        **kwargs: Any,
    ) -> BeamSearchResult:
        loop_state = init_loop_state(solve_config, start, **kwargs)
        loop_state = jax.lax.while_loop(loop_condition, loop_body, loop_state)
        search_result = loop_state.search_result
        filled_mask = search_result.filled_mask()
        solved_mask = puzzle.batched_is_solved(solve_config, search_result.beam)
        solved_mask = jnp.logical_and(solved_mask, filled_mask)

        solved_any = solved_mask.any()
        solved_idx = jnp.argmax(solved_mask)
        solved_idx = jnp.where(
            solved_any, solved_idx.astype(jnp.int32), jnp.array(-1, dtype=jnp.int32)
        )

        search_result.solved = solved_any
        search_result.solved_idx = solved_idx
        return search_result

    beam_fn = jax.jit(beam)
    if show_compile_time:
        print("initializing jit")
        start = time.time()

    if warmup_inputs is None:
        empty_solve_config = puzzle.SolveConfig.default()
        empty_states = puzzle.State.default()
        # Compile ahead of time to surface potential tracer issues early. This uses
        # empty defaults, mirroring the A*/Q* builders.
        beam_fn(empty_solve_config, empty_states)
    else:
        compile_with_example(beam_fn, *warmup_inputs)

    if show_compile_time:
        end = time.time()
        compile_time = end - start
        print("Compile Time: {:6.2f} seconds".format(compile_time))
        print("JIT compiled\n\n")

    return beam_fn
