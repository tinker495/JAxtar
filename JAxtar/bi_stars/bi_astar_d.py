"""
JAxtar Bidirectional A* Deferred Search Implementation

This module implements bidirectional A* with deferred heuristic evaluation.
Like A* deferred, heuristic evaluation is delayed until nodes are popped from
the priority queue, combined with bidirectional search for maximum efficiency.

Key Benefits:
- Reduced heuristic evaluations (only evaluated when popped)
- Bidirectional search reduces search space
- Efficient for expensive heuristics with large branching factors
"""

from typing import Any

import chex
import jax
import jax.numpy as jnp
from puxle import Puzzle

from helpers.jax_compile import compile_search_builder
from heuristic.heuristic_base import Heuristic
from JAxtar.annotate import KEY_DTYPE, MIN_BATCH_SIZE
from JAxtar.bi_stars.bi_search_base import (
    BiDirectionalSearchResult,
    BiLoopStateWithStates,
    build_bi_search_result,
    common_bi_loop_condition,
    initialize_bi_loop_common,
    materialize_meeting_point_hashidxs,
    build_bi_deferred_expand_direction,
)
from JAxtar.stars.search_base import (
    Current,
)
from JAxtar.utils.array_ops import stable_partition_three
from JAxtar.utils.batch_switcher import variable_batch_switcher_builder


def _bi_astar_d_loop_builder(
    puzzle: Puzzle,
    heuristic: Heuristic,
    bi_result_template: BiDirectionalSearchResult,
    batch_size: int = 1024,
    cost_weight: float = 1.0 - 1e-6,
    look_ahead_pruning: bool = True,
    use_backward_heuristic: bool = True,
    backward_value_lookahead: bool = True,
    backward_value_lookahead_k: int | None = None,
    terminate_on_first_solution: bool = False,
):
    """
    Build the loop components for bidirectional A* deferred search.

    Args:
        puzzle: Puzzle instance
        heuristic: Heuristic instance (used for both directions)
        bi_result_template: Pre-built BiDirectionalSearchResult template
        batch_size: Batch size for parallel processing
        cost_weight: Weight for path cost in f = cost_weight * g + h
        look_ahead_pruning: Enable look-ahead pruning optimization

    Returns:
        Tuple of (init_loop_state, loop_condition, loop_body) functions
    """
    action_size = puzzle.action_size

    variable_heuristic_batch_switcher = variable_batch_switcher_builder(
        heuristic.batched_distance,
        max_batch_size=batch_size,
        min_batch_size=MIN_BATCH_SIZE,
        pad_value=jnp.inf,
    )

    def init_loop_state(
        bi_result: BiDirectionalSearchResult,
        solve_config: Puzzle.SolveConfig,
        inverse_solveconfig: Puzzle.SolveConfig,
        start: Puzzle.State,
        heuristic_params_forward: Any,
        heuristic_params_backward: Any,
    ) -> BiLoopStateWithStates:
        """Initialize bidirectional deferred search from start and goal states."""

        (
            fwd_filled,
            fwd_current,
            fwd_states,
            bwd_filled,
            bwd_current,
            bwd_states,
        ) = initialize_bi_loop_common(bi_result, puzzle, solve_config, start)

        return BiLoopStateWithStates(
            bi_result=bi_result,
            solve_config=solve_config,
            inverse_solveconfig=inverse_solveconfig,
            params_forward=heuristic_params_forward,
            params_backward=heuristic_params_backward,
            current_forward=fwd_current,
            current_backward=bwd_current,
            states_forward=fwd_states,
            states_backward=bwd_states,
            filled_forward=fwd_filled,
            filled_backward=bwd_filled,
        )

    def loop_condition(loop_state: BiLoopStateWithStates) -> chex.Array:
        """Check if search should continue."""
        return common_bi_loop_condition(
            loop_state.bi_result,
            loop_state.filled_forward,
            loop_state.filled_backward,
            loop_state.current_forward,
            loop_state.current_backward,
            cost_weight,
            terminate_on_first_solution,
        )

    def _eval_deferred_heuristic(
        is_forward: bool,
        solve_config: Puzzle.SolveConfig,
        inverse_solveconfig: Puzzle.SolveConfig,
        heuristic_params: Any,
        states: Puzzle.State,
        costs: chex.Array,
        look_a_head_costs: chex.Array,
        ncosts: chex.Array,
        filled: chex.Array,
        filled_tiles: chex.Array,
        optimal_mask: chex.Array,
        found: chex.Array,
        current_hash_idxs: chex.Array,
        search_result: Any,
        neighbour_look_a_head: Puzzle.State,
        use_heuristic: bool,
    ) -> tuple[chex.Array, chex.Array]:
        sr_batch_size = search_result.batch_size
        flat_size = action_size * sr_batch_size
        flattened_neighbour_look_head = neighbour_look_a_head.flatten()

        if look_ahead_pruning:
            if use_heuristic:
                found_reshaped = found.reshape(action_size, sr_batch_size)
                optimal_mask_reshaped = optimal_mask.reshape(action_size, sr_batch_size)

                old_dists = search_result.get_dist(current_hash_idxs)
                old_dists = old_dists.reshape(action_size, sr_batch_size)

                need_compute = optimal_mask_reshaped & ~found_reshaped
                flat_states = neighbour_look_a_head.flatten()
                flat_need_compute = need_compute.flatten()

                n = flat_size
                sorted_indices = stable_partition_three(
                    flat_need_compute, jnp.zeros_like(flat_need_compute, dtype=jnp.bool_)
                )

                sorted_states = flat_states[sorted_indices]
                sorted_mask = flat_need_compute[sorted_indices]

                sorted_states_chunked = sorted_states.reshape((action_size, sr_batch_size))
                sorted_mask_chunked = sorted_mask.reshape((action_size, sr_batch_size))

                def _calc_heuristic_chunk(carry, input_slice):
                    states_slice, compute_mask = input_slice
                    h_val = variable_heuristic_batch_switcher(
                        heuristic_params, states_slice, compute_mask
                    )
                    return carry, h_val

                _, h_val_chunks = jax.lax.scan(
                    _calc_heuristic_chunk,
                    None,
                    (sorted_states_chunked, sorted_mask_chunked),
                )

                h_val_sorted = h_val_chunks.reshape(-1)
                flat_h_val = (
                    jnp.empty((n,), dtype=h_val_sorted.dtype).at[sorted_indices].set(h_val_sorted)
                )
                computed_heuristic_vals = flat_h_val.reshape(action_size, sr_batch_size)

                heuristic_vals = jnp.where(
                    found_reshaped,
                    old_dists,
                    computed_heuristic_vals,
                )
                heuristic_vals = jnp.where(filled_tiles, heuristic_vals, jnp.inf).astype(KEY_DTYPE)
            else:
                heuristic_vals = jnp.zeros_like(look_a_head_costs, dtype=KEY_DTYPE)
                heuristic_vals = jnp.where(filled_tiles, heuristic_vals, jnp.inf)

            if (not is_forward) and use_heuristic and backward_value_lookahead:
                base_keys = (cost_weight * look_a_head_costs + heuristic_vals).astype(KEY_DTYPE)
                flat_keys = base_keys.flatten()
                masked_flat_keys = jnp.where(optimal_mask, flat_keys, jnp.inf)

                sort_idx = jnp.arange(flat_size, dtype=jnp.int32)
                sorted_keys, sorted_idx = jax.lax.sort_key_val(masked_flat_keys, sort_idx)

                k = (
                    sr_batch_size
                    if backward_value_lookahead_k is None
                    else backward_value_lookahead_k
                )
                k = min(k, sr_batch_size, flat_size)
                sel_idx = sorted_idx[:k]
                sel_valid = jnp.isfinite(sorted_keys[:k])

                sel_states = flattened_neighbour_look_head[sel_idx]
                succ_states, succ_costs = puzzle.batched_get_neighbours(
                    solve_config, sel_states, sel_valid
                )

                def _scan_backup(carry, inputs):
                    succ_slice, cost_slice = inputs
                    h_succ = variable_heuristic_batch_switcher(
                        heuristic_params, succ_slice, sel_valid
                    ).astype(KEY_DTYPE)
                    q_est = h_succ + cost_slice.astype(KEY_DTYPE)
                    return carry, q_est

                _, q_chunks = jax.lax.scan(_scan_backup, None, (succ_states, succ_costs))
                v_sel = jnp.min(q_chunks, axis=0).astype(KEY_DTYPE)

                heur_flat = heuristic_vals.flatten()
                updated = jnp.where(sel_valid, v_sel, heur_flat[sel_idx])
                heur_flat = heur_flat.at[sel_idx].set(updated)
                heuristic_vals = heur_flat.reshape((action_size, sr_batch_size))

            neighbour_keys = (cost_weight * look_a_head_costs + heuristic_vals).astype(KEY_DTYPE)

        else:
            if use_heuristic:
                heuristic_vals = variable_heuristic_batch_switcher(heuristic_params, states, filled)
                heuristic_vals = jnp.where(filled, heuristic_vals, jnp.inf)
                heuristic_vals = jnp.tile(heuristic_vals[jnp.newaxis, :], (action_size, 1)).astype(
                    KEY_DTYPE
                )
            else:
                heuristic_vals = jnp.zeros_like(costs, dtype=KEY_DTYPE)
                heuristic_vals = jnp.where(filled_tiles, heuristic_vals, jnp.inf)

            neighbour_keys = (cost_weight * costs + heuristic_vals).astype(KEY_DTYPE)

        return heuristic_vals, neighbour_keys

    _expand_direction_deferred = build_bi_deferred_expand_direction(
        puzzle, cost_weight, look_ahead_pruning, _eval_deferred_heuristic
    )

    def loop_body(loop_state: BiLoopStateWithStates) -> BiLoopStateWithStates:
        """Main loop body for bidirectional A* deferred."""
        bi_result = loop_state.bi_result
        solve_config = loop_state.solve_config
        inverse_solveconfig = loop_state.inverse_solveconfig

        fwd_not_full = bi_result.forward.generated_size < bi_result.forward.capacity
        bwd_not_full = bi_result.backward.generated_size < bi_result.backward.capacity

        def _expand_forward(bi_result):
            return _expand_direction_deferred(
                bi_result,
                solve_config,
                inverse_solveconfig,
                loop_state.params_forward,
                loop_state.current_forward,
                loop_state.states_forward,
                loop_state.filled_forward,
                True,  # is_forward
                True,  # use_heuristic
            )

        def _expand_backward(bi_result):
            return _expand_direction_deferred(
                bi_result,
                solve_config,
                inverse_solveconfig,
                loop_state.params_backward,
                loop_state.current_backward,
                loop_state.states_backward,
                loop_state.filled_backward,
                False,  # is_forward
                use_backward_heuristic,  # use_heuristic
            )

        # Expand both directions
        bi_result, new_fwd_current, new_fwd_states, new_fwd_filled = jax.lax.cond(
            jnp.logical_and(loop_state.filled_forward.any(), fwd_not_full),
            _expand_forward,
            lambda br: (
                br,
                loop_state.current_forward,
                loop_state.states_forward,
                loop_state.filled_forward,
            ),
            bi_result,
        )

        bi_result, new_bwd_current, new_bwd_states, new_bwd_filled = jax.lax.cond(
            jnp.logical_and(loop_state.filled_backward.any(), bwd_not_full),
            _expand_backward,
            lambda br: (
                br,
                loop_state.current_backward,
                loop_state.states_backward,
                loop_state.filled_backward,
            ),
            bi_result,
        )

        return BiLoopStateWithStates(
            bi_result=bi_result,
            solve_config=solve_config,
            inverse_solveconfig=inverse_solveconfig,
            params_forward=loop_state.params_forward,
            params_backward=loop_state.params_backward,
            current_forward=new_fwd_current,
            current_backward=new_bwd_current,
            states_forward=new_fwd_states,
            states_backward=new_bwd_states,
            filled_forward=new_fwd_filled,
            filled_backward=new_bwd_filled,
        )

    return init_loop_state, loop_condition, loop_body


def bi_astar_d_builder(
    puzzle: Puzzle,
    heuristic: Heuristic,
    batch_size: int = 1024,
    max_nodes: int = int(1e6),
    pop_ratio: float = jnp.inf,
    cost_weight: float = 1.0 - 1e-6,
    show_compile_time: bool = False,
    look_ahead_pruning: bool = True,
    terminate_on_first_solution: bool = True,
    warmup_inputs: tuple[Puzzle.SolveConfig, Puzzle.State] | None = None,
):
    """
    Builds and returns a JAX-accelerated bidirectional A* deferred search function.

    Combines bidirectional search with deferred heuristic evaluation for
    maximum efficiency when heuristics are expensive and branching factor is large.

    Args:
        puzzle: Puzzle instance (must support batched_get_inverse_neighbours)
        heuristic: Heuristic instance
        batch_size: Number of states to process in parallel per direction
        max_nodes: Maximum number of nodes to explore per direction
        pop_ratio: Ratio controlling beam width
        cost_weight: Weight for path cost in f = cost_weight * g + h
        show_compile_time: If True, displays compilation time
        look_ahead_pruning: Enable look-ahead pruning optimization.
            Note: Bidirectional deferred search requires look_ahead_pruning=True
            for correct termination condition. If False is passed, it will be
            forced to True with a warning.

    Returns:
        A JIT-compiled function that performs bidirectional A* deferred search
    """
    # Bidirectional deferred requires look_ahead_pruning for correct f-value computation
    # Without it, the termination condition based on get_min_f_value() can be incorrect
    if not look_ahead_pruning:
        import warnings

        warnings.warn(
            "Bidirectional A* deferred requires look_ahead_pruning=True for correct "
            "termination. Forcing look_ahead_pruning=True.",
            UserWarning,
        )
        look_ahead_pruning = True

    statecls = puzzle.State
    action_size = puzzle.action_size
    denom = max(1, puzzle.action_size // 2)
    min_pop = max(1, MIN_BATCH_SIZE // denom)

    # Pre-build the search result OUTSIDE of JIT context
    bi_result_template = build_bi_search_result(
        statecls,
        batch_size,
        max_nodes,
        action_size,
        pop_ratio=pop_ratio,
        min_pop=min_pop,
        parant_with_costs=True,
    )

    use_backward_heuristic = not heuristic.is_fixed
    init_loop_state, loop_condition, loop_body = _bi_astar_d_loop_builder(
        puzzle,
        heuristic,
        bi_result_template,
        batch_size,
        cost_weight,
        look_ahead_pruning,
        use_backward_heuristic=use_backward_heuristic,
        terminate_on_first_solution=terminate_on_first_solution,
    )

    def bi_astar_d(
        solve_config: Puzzle.SolveConfig,
        start: Puzzle.State,
        **kwargs: Any,
    ) -> BiDirectionalSearchResult:
        """Perform bidirectional A* deferred search."""
        # Prepare heuristic parameters for both directions
        heuristic_params_forward = heuristic.prepare_heuristic_parameters(solve_config, **kwargs)
        # Build a backward solve config that treats `start` as the target.
        # Prefer puzzle-level normalization via hindsight_transform.
        inverse_solveconfig = puzzle.hindsight_transform(solve_config, start)

        if use_backward_heuristic:
            heuristic_params_backward = heuristic.prepare_heuristic_parameters(
                inverse_solveconfig, **kwargs
            )
        else:
            heuristic_params_backward = heuristic_params_forward

        loop_state = init_loop_state(
            bi_result_template,
            solve_config,
            inverse_solveconfig,
            start,
            heuristic_params_forward,
            heuristic_params_backward,
        )
        loop_state = jax.lax.while_loop(loop_condition, loop_body, loop_state)

        bi_result = loop_state.bi_result

        # Materialize meeting hashidxs if the best meeting was found via edge-only tracking.
        bi_result = materialize_meeting_point_hashidxs(bi_result, puzzle, solve_config)

        # Mark as solved if meeting point was found
        bi_result.forward.solved = bi_result.meeting.found
        bi_result.forward.solved_idx = Current(
            hashidx=bi_result.meeting.fwd_hashidx,
            cost=bi_result.meeting.fwd_cost,
        )
        bi_result.backward.solved = bi_result.meeting.found
        bi_result.backward.solved_idx = Current(
            hashidx=bi_result.meeting.bwd_hashidx,
            cost=bi_result.meeting.bwd_cost,
        )

        return bi_result

    return compile_search_builder(bi_astar_d, puzzle, show_compile_time, warmup_inputs)
