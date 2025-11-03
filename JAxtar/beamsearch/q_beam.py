import time

import jax
import jax.numpy as jnp
import xtructure.numpy as xnp
from puxle import Puzzle

from JAxtar.annotate import ACTION_DTYPE, KEY_DTYPE, MIN_BATCH_SIZE
from JAxtar.beamsearch.search_base import (
    ACTION_PAD,
    TRACE_INDEX_DTYPE,
    TRACE_INVALID,
    BeamSearchResult,
    select_beam,
)
from JAxtar.utils.batch_switcher import variable_batch_switcher_builder
from qfunction.q_base import QFunction


def qbeam_builder(
    puzzle: Puzzle,
    q_fn: QFunction,
    batch_size: int = 1024,
    max_nodes: int = int(1e6),
    pop_ratio: float = jnp.inf,
    cost_weight: float = 1.0 - 1e-6,
    show_compile_time: bool = False,
):
    """Construct a batched Q*-style beam search solver without hash tables."""

    statecls = puzzle.State
    beam_width = batch_size
    denom = max(1, puzzle.action_size // 2)
    min_keep = max(1, beam_width // denom)
    pop_ratio = float(pop_ratio)
    max_depth = max(1, (max_nodes + beam_width - 1) // beam_width)
    variable_q_batch_switcher = variable_batch_switcher_builder(
        lambda solve_config, current: q_fn.batched_q_value(solve_config, current),
        max_batch_size=beam_width,
        min_batch_size=MIN_BATCH_SIZE,
        pad_value=jnp.inf,
    )

    def qbeam(
        solve_config: Puzzle.SolveConfig,
        start: Puzzle.State,
    ) -> BeamSearchResult:
        result = BeamSearchResult.build(
            statecls,
            beam_width,
            max_depth,
        )

        result.beam = result.beam.at[0].set(start)
        result.cost = result.cost.at[0].set(0)
        result.dist = result.dist.at[0].set(0)
        result.scores = result.scores.at[0].set(0)
        result.parent_index = result.parent_index.at[0].set(-1)
        result.active_trace = result.active_trace.at[0].set(0)
        result.trace_cost = result.trace_cost.at[0].set(0)
        result.trace_dist = result.trace_dist.at[0].set(0)
        result.trace_depth = result.trace_depth.at[0].set(0)
        result.trace_action = result.trace_action.at[0].set(ACTION_PAD)

        def _cond(search_result: BeamSearchResult):
            filled_mask = search_result.filled_mask()
            has_states = filled_mask.any()
            depth_ok = search_result.depth < max_depth

            beam_states = search_result.beam
            solved = puzzle.batched_is_solved(solve_config, beam_states)
            solved = jnp.logical_and(solved, filled_mask)
            return jnp.logical_and(jnp.logical_and(depth_ok, has_states), ~solved.any())

        def _body(search_result: BeamSearchResult):
            filled_mask = search_result.filled_mask()
            beam_states = search_result.beam

            neighbours, transition_cost = puzzle.batched_get_neighbours(
                solve_config, beam_states, filled_mask
            )

            base_costs = search_result.cost
            child_costs = (base_costs[jnp.newaxis, :] + transition_cost).astype(KEY_DTYPE)
            child_valid = jnp.logical_and(filled_mask[jnp.newaxis, :], jnp.isfinite(child_costs))
            child_costs = jnp.where(child_valid, child_costs, jnp.inf)

            def _compute_q(_):
                vals = variable_q_batch_switcher(solve_config, beam_states, filled_mask)
                vals = vals.transpose().astype(KEY_DTYPE)
                return jnp.where(child_valid, vals, jnp.inf)

            q_vals = jax.lax.cond(
                jnp.any(child_valid),
                _compute_q,
                lambda _: jnp.full(child_valid.shape, jnp.inf, dtype=KEY_DTYPE),
                None,
            )

            scores = (cost_weight * child_costs + q_vals).astype(KEY_DTYPE)
            scores = jnp.where(child_valid, scores, jnp.inf)

            flat_states = neighbours.flatten()
            flat_cost = child_costs.reshape(-1)
            flat_q = q_vals.reshape(-1)
            flat_scores = scores.reshape(-1)
            flat_valid = child_valid.reshape(-1)

            selected_scores, selected_idx, keep_mask = select_beam(
                flat_scores,
                beam_width,
                pop_ratio=pop_ratio,
                min_keep=min_keep,
            )

            selected_states = flat_states[selected_idx]
            selected_costs = flat_cost[selected_idx]
            selected_q = flat_q[selected_idx]
            selected_actions = (selected_idx // beam_width).astype(ACTION_DTYPE)
            selected_parents = (selected_idx % beam_width).astype(jnp.int32)
            selected_valid = jnp.logical_and(keep_mask, flat_valid[selected_idx])
            unique_valid = xnp.unique_mask(
                selected_states,
                key=selected_scores,
                filled=selected_valid,
            )
            selected_valid = jnp.logical_and(selected_valid, unique_valid)

            selected_costs = jnp.where(selected_valid, selected_costs, jnp.inf)
            selected_q = jnp.where(selected_valid, selected_q, jnp.inf)
            selected_scores = jnp.where(selected_valid, selected_scores, jnp.inf)
            selected_actions = selected_actions.astype(ACTION_DTYPE)

            parent_trace_ids = search_result.active_trace[selected_parents]
            invalid_parent = jnp.full_like(parent_trace_ids, TRACE_INVALID)
            parent_trace_ids = jnp.where(selected_valid, parent_trace_ids, invalid_parent)

            next_depth_idx = jnp.minimum(search_result.depth + 1, max_depth)
            trace_offset = next_depth_idx.astype(TRACE_INDEX_DTYPE) * jnp.asarray(
                beam_width, dtype=TRACE_INDEX_DTYPE
            )
            slot_indices = jnp.arange(beam_width, dtype=TRACE_INDEX_DTYPE)
            next_trace_ids = trace_offset + slot_indices

            trace_actions = jnp.where(
                selected_valid,
                selected_actions,
                jnp.full_like(selected_actions, ACTION_PAD),
            )
            depth_fill = jnp.full((beam_width,), next_depth_idx, dtype=jnp.int32)
            depth_default = -jnp.ones((beam_width,), dtype=jnp.int32)
            trace_depths = jnp.where(selected_valid, depth_fill, depth_default)

            search_result.trace_parent = search_result.trace_parent.at[next_trace_ids].set(
                parent_trace_ids
            )
            search_result.trace_action = search_result.trace_action.at[next_trace_ids].set(
                trace_actions
            )
            search_result.trace_cost = search_result.trace_cost.at[next_trace_ids].set(
                selected_costs
            )
            search_result.trace_dist = search_result.trace_dist.at[next_trace_ids].set(selected_q)
            search_result.trace_depth = search_result.trace_depth.at[next_trace_ids].set(
                trace_depths
            )

            invalid_trace = jnp.full_like(next_trace_ids, TRACE_INVALID)
            next_active_trace = jnp.where(selected_valid, next_trace_ids, invalid_trace)

            search_result.beam = selected_states
            search_result.cost = selected_costs
            search_result.dist = selected_q
            search_result.scores = selected_scores
            search_result.parent_index = jnp.where(
                selected_valid, selected_parents, -jnp.ones_like(selected_parents)
            )
            search_result.active_trace = next_active_trace
            selected_count = selected_valid.astype(jnp.int32).sum()
            search_result.generated_size = search_result.generated_size + selected_count
            search_result.depth = search_result.depth + 1
            return search_result

        search_result = jax.lax.while_loop(_cond, _body, result)

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

    qbeam_fn = jax.jit(qbeam)
    empty_solve_config = puzzle.SolveConfig.default()
    empty_states = puzzle.State.default()

    if show_compile_time:
        print("initializing jit")
        start = time.time()

    qbeam_fn(empty_solve_config, empty_states)

    if show_compile_time:
        end = time.time()
        print(f"Compile Time: {end - start:6.2f} seconds")
        print("JIT compiled\n\n")

    return qbeam_fn


__all__ = ["qbeam_builder"]
