import jax
import jax.numpy as jnp
import xtructure.numpy as xnp
from puxle import Puzzle

from helpers.jax_compile import jit_with_warmup
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
from JAxtar.utils.batch_switcher import variable_batch_switcher_builder
from qfunction.q_base import QFunction


def _qbeam_loop_builder(
    puzzle: Puzzle,
    q_fn: QFunction,
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
    variable_q_batch_switcher = variable_batch_switcher_builder(
        q_fn.batched_q_value,
        max_batch_size=beam_width,
        min_batch_size=MIN_BATCH_SIZE,
        pad_value=jnp.inf,
    )

    if non_backtracking_steps < 0:
        raise ValueError("non_backtracking_steps must be non-negative")
    non_backtracking_steps = int(non_backtracking_steps)

    def init_loop_state(solve_config: Puzzle.SolveConfig, start: Puzzle.State):
        result = BeamSearchResult.build(
            statecls,
            beam_width,
            max_depth,
            action_size,
        )
        q_parameters = q_fn.prepare_q_parameters(solve_config)

        # Seed the beam with the start state
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
        result.trace_state = result.trace_state.at[0].set(start)

        return BeamSearchLoopState(
            search_result=result,
            solve_config=solve_config,
            params=q_parameters,
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
        q_parameters = loop_state.params

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

        def _compute_q(_):
            vals = variable_q_batch_switcher(q_parameters, beam_states, filled_mask)
            vals = vals.transpose().astype(KEY_DTYPE)
            return jnp.where(child_valid, vals, jnp.inf)

        q_vals = (
            jax.lax.cond(
                jnp.any(child_valid),
                _compute_q,
                lambda _: jnp.full(child_shape, jnp.inf, dtype=KEY_DTYPE),
                None,
            )
            - transition_cost
        )  # Q(s,a) = h(s') + c(s,a) / h(s') = Q(s,a) - c(s,a)

        scores = (cost_weight * child_costs + q_vals).astype(KEY_DTYPE)
        scores = jnp.where(child_valid, scores, jnp.inf)

        flat_states = neighbours.reshape((flat_count,))
        flat_cost = child_costs.reshape((flat_count,))
        flat_q = q_vals.reshape((flat_count,))
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
        selected_q = flat_q[selected_idx]
        selected_actions = (selected_idx // sr_beam_width).astype(ACTION_DTYPE)
        selected_parents = (selected_idx % sr_beam_width).astype(jnp.int32)
        selected_valid = jnp.logical_and(keep_mask, flat_valid[selected_idx])
        unique_valid = xnp.unique_mask(
            selected_states,
            key=selected_scores,
            filled=selected_valid,
        )
        selected_valid = jnp.logical_and(selected_valid, unique_valid)

        if non_backtracking_steps:
            allowed_mask = non_backtracking_mask(
                selected_states,
                search_result.active_trace[selected_parents],
                search_result.trace_state,
                search_result.trace_parent,
                non_backtracking_steps,
            )
            selected_valid = jnp.logical_and(selected_valid, allowed_mask)

        search_result = apply_selected_candidates(
            search_result,
            selected_states,
            selected_costs,
            selected_q,
            selected_scores,
            selected_actions,
            selected_parents,
            selected_valid,
        )

        return BeamSearchLoopState(
            search_result=search_result,
            solve_config=solve_config,
            params=q_parameters,
        )

    return init_loop_state, loop_condition, loop_body


def qbeam_builder(
    puzzle: Puzzle,
    q_fn: QFunction,
    batch_size: int = 1024,
    max_nodes: int = int(1e6),
    pop_ratio: float = jnp.inf,
    cost_weight: float = 1.0 - 1e-6,
    show_compile_time: bool = False,
    non_backtracking_steps: int = 3,
    warmup_inputs: tuple[Puzzle.SolveConfig, Puzzle.State] | None = None,
):
    """Construct a batched Q*-style beam search solver without hash tables."""

    (init_loop_state, loop_condition, loop_body,) = _qbeam_loop_builder(
        puzzle,
        q_fn,
        batch_size,
        max_nodes,
        pop_ratio,
        cost_weight,
        non_backtracking_steps,
    )

    def qbeam(
        solve_config: Puzzle.SolveConfig,
        start: Puzzle.State,
    ) -> BeamSearchResult:
        loop_state = init_loop_state(solve_config, start)
        loop_state = jax.lax.while_loop(loop_condition, loop_body, loop_state)
        return finalize_beam_search_result(
            loop_state.search_result,
            puzzle,
            solve_config,
        )

    return jit_with_warmup(
        qbeam,
        puzzle=puzzle,
        show_compile_time=show_compile_time,
        warmup_inputs=warmup_inputs,
    )


__all__ = ["qbeam_builder"]
