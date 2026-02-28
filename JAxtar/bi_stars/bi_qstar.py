"""
JAxtar Bidirectional Q* Search Implementation

This module implements bidirectional Q* search, which uses Q-functions (state-action
value functions) instead of heuristics, combined with bidirectional search for
exploring from both start and goal simultaneously.

Key Benefits:
- Q-functions provide action-dependent value estimates
- Bidirectional search reduces search space
- Efficient for domains where Q-learning has been applied
"""

from typing import Any

import chex
import jax
import jax.numpy as jnp
from puxle import Puzzle

from helpers.jax_compile import compile_search_builder
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
from qfunction.q_base import QFunction


def _bi_qstar_loop_builder(
    puzzle: Puzzle,
    q_fn: QFunction,
    bi_result_template: BiDirectionalSearchResult,
    batch_size: int = 1024,
    cost_weight: float = 1.0 - 1e-6,
    look_ahead_pruning: bool = True,
    pessimistic_update: bool = True,
    use_backward_q: bool = True,
    backward_mode: str = "auto",
    terminate_on_first_solution: bool = False,
):
    """
    Build the loop components for bidirectional Q* search.

    Args:
        puzzle: Puzzle instance
        q_fn: QFunction instance (used for both directions)
        bi_result_template: Pre-built BiDirectionalSearchResult template
        batch_size: Batch size for parallel processing
        cost_weight: Weight for path cost in f = cost_weight * g + Q(s,a)
        look_ahead_pruning: Enable look-ahead pruning optimization
        pessimistic_update: Use max Q-value for duplicates (True) or min (False)

    Returns:
        Tuple of (init_loop_state, loop_condition, loop_body) functions
    """
    action_size = puzzle.action_size

    variable_q_batch_switcher = variable_batch_switcher_builder(
        q_fn.batched_q_value,
        max_batch_size=batch_size,
        min_batch_size=MIN_BATCH_SIZE,
        pad_value=jnp.inf,
    )

    def init_loop_state(
        bi_result: BiDirectionalSearchResult,
        solve_config: Puzzle.SolveConfig,
        inverse_solveconfig: Puzzle.SolveConfig,
        start: Puzzle.State,
        q_params_forward: Any,
        q_params_backward: Any,
    ) -> BiLoopStateWithStates:
        """Initialize bidirectional Q* search from start and goal states."""

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
            params_forward=q_params_forward,
            params_backward=q_params_backward,
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

    def _eval_deferred_q(
        is_forward: bool,
        solve_config: Puzzle.SolveConfig,
        inverse_solveconfig: Puzzle.SolveConfig,
        q_params: Any,
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
        use_q: bool,
    ) -> tuple[chex.Array, chex.Array]:
        sr_batch_size = search_result.batch_size
        flat_size = action_size * sr_batch_size

        if (not is_forward) and use_q:
            if backward_mode == "auto":
                can_optimize_bwd = hasattr(puzzle, "inverse_action_map")
                use_value_heuristic = not can_optimize_bwd
            elif backward_mode == "edge_q":
                can_optimize_bwd = hasattr(puzzle, "inverse_action_map")
                use_value_heuristic = False
            elif backward_mode == "value_v":
                can_optimize_bwd = False
                use_value_heuristic = True
            else:
                can_optimize_bwd = False
                use_value_heuristic = False
        else:
            can_optimize_bwd = False
            use_value_heuristic = False

        flattened_filled_tiles = filled_tiles.flatten()

        if use_value_heuristic:
            flattened_neighbour_look_head = neighbour_look_a_head.flatten()

            flat_states = flattened_neighbour_look_head
            flat_mask = flattened_filled_tiles

            n = flat_size
            invperm = stable_partition_three(flat_mask, jnp.zeros_like(flat_mask, dtype=jnp.bool_))
            sorted_states = flat_states[invperm]
            sorted_mask = flat_mask[invperm]

            sorted_states_chunked = sorted_states.reshape((action_size, sr_batch_size))
            sorted_mask_chunked = sorted_mask.reshape((action_size, sr_batch_size))

            def _calc_v_chunk(carry, input_slice):
                states_slice, compute_mask = input_slice
                q_vals = variable_q_batch_switcher(q_params, states_slice, compute_mask)
                v_vals = jnp.min(q_vals, axis=-1)
                return carry, v_vals

            _, v_chunks = jax.lax.scan(
                _calc_v_chunk,
                None,
                (sorted_states_chunked, sorted_mask_chunked),
            )

            v_sorted = v_chunks.reshape(-1).astype(KEY_DTYPE)
            v_flat = jnp.empty((n,), dtype=v_sorted.dtype).at[invperm].set(v_sorted)

            heuristic_vals = v_flat.reshape(action_size, sr_batch_size)
            heuristic_vals = jnp.where(filled_tiles, heuristic_vals, jnp.inf).astype(KEY_DTYPE)

            neighbour_keys = (cost_weight * look_a_head_costs + heuristic_vals).astype(KEY_DTYPE)
        else:
            if use_q:
                q_vals = variable_q_batch_switcher(q_params, states, filled)
                q_vals = q_vals.transpose().astype(KEY_DTYPE)

                if can_optimize_bwd:
                    inv_map = puzzle.inverse_action_map
                    q_vals = q_vals[inv_map, :]
            else:
                q_vals = ncosts.astype(KEY_DTYPE)

            neighbour_keys = (cost_weight * costs + q_vals).astype(KEY_DTYPE)
            raw_q_flat = q_vals.flatten()
            dists = raw_q_flat

            if look_ahead_pruning:
                old_dists = search_result.get_dist(current_hash_idxs)

                if use_q:
                    q_old_reconstructed = old_dists.astype(KEY_DTYPE) + ncosts.flatten().astype(
                        KEY_DTYPE
                    )
                    if pessimistic_update:
                        q_old_for_max = jnp.where(found, q_old_reconstructed, -jnp.inf)
                        dists = jnp.maximum(dists, q_old_for_max)
                    else:
                        q_old_for_min = jnp.where(found, q_old_reconstructed, jnp.inf)
                        dists = jnp.minimum(dists, q_old_for_min)

            heuristic_vals = dists.reshape(action_size, sr_batch_size)

        neighbour_keys = jnp.where(filled_tiles, neighbour_keys, jnp.inf)
        return heuristic_vals, neighbour_keys

    def override_use_heuristic_in_pop(is_forward: bool, use_q: bool) -> bool:
        if (not is_forward) and use_q:
            if backward_mode == "auto":
                can_optimize_bwd = hasattr(puzzle, "inverse_action_map")
                use_value_heuristic = not can_optimize_bwd
            elif backward_mode == "edge_q":
                can_optimize_bwd = hasattr(puzzle, "inverse_action_map")
                use_value_heuristic = False
            elif backward_mode == "value_v":
                can_optimize_bwd = False
                use_value_heuristic = True
            else:
                use_value_heuristic = False
            return use_value_heuristic

        return False

    _eval_deferred_q.override_use_heuristic_in_pop = override_use_heuristic_in_pop

    _expand_direction_q = build_bi_deferred_expand_direction(
        puzzle, cost_weight, look_ahead_pruning, _eval_deferred_q
    )

    def loop_body(loop_state: BiLoopStateWithStates) -> BiLoopStateWithStates:
        """Main loop body for bidirectional Q*."""
        bi_result = loop_state.bi_result
        solve_config = loop_state.solve_config
        inverse_solveconfig = loop_state.inverse_solveconfig

        fwd_not_full = bi_result.forward.generated_size < bi_result.forward.capacity
        bwd_not_full = bi_result.backward.generated_size < bi_result.backward.capacity

        def _expand_forward(bi_result):
            return _expand_direction_q(
                bi_result,
                solve_config,
                inverse_solveconfig,
                loop_state.params_forward,
                loop_state.current_forward,
                loop_state.states_forward,
                loop_state.filled_forward,
                use_evaluation=True,
                is_forward=True,
            )

        def _expand_backward(bi_result):
            return _expand_direction_q(
                bi_result,
                solve_config,
                inverse_solveconfig,
                loop_state.params_backward,
                loop_state.current_backward,
                loop_state.states_backward,
                loop_state.filled_backward,
                use_evaluation=use_backward_q,
                is_forward=False,
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


def bi_qstar_builder(
    puzzle: Puzzle,
    q_fn: QFunction,
    batch_size: int = 1024,
    max_nodes: int = int(1e6),
    pop_ratio: float = jnp.inf,
    cost_weight: float = 1.0 - 1e-6,
    show_compile_time: bool = False,
    look_ahead_pruning: bool = True,
    pessimistic_update: bool = True,
    backward_mode: str = "auto",
    terminate_on_first_solution: bool = True,
    warmup_inputs: tuple[Puzzle.SolveConfig, Puzzle.State] | None = None,
):
    """
    Builds and returns a JAX-accelerated bidirectional Q* search function.

    Combines bidirectional search with Q-function evaluation for efficient
    search in domains where Q-learning has been applied.

    Args:
        puzzle: Puzzle instance (must support batched_get_inverse_neighbours)
        q_fn: QFunction instance for state-action value estimation
        batch_size: Number of states to process in parallel per direction
        max_nodes: Maximum number of nodes to explore per direction
        pop_ratio: Ratio controlling beam width
        cost_weight: Weight for path cost in f = cost_weight * g + Q(s,a)
        show_compile_time: If True, displays compilation time
        look_ahead_pruning: Enable look-ahead pruning optimization
        pessimistic_update: Use max Q-value for duplicates (True) or min (False)
        backward_mode: Backward-direction scoring mode:
            - "auto": use "edge_q" if puzzle.inverse_action_map exists, else "value_v".
            - "edge_q": use Q(parent, a) for backward edges via inverse_action_map.
            - "value_v": use V(s)=min_a Q(s,a) as a heuristic on predecessor states.
            - "dijkstra": ignore Q in backward direction and use true step costs.
        terminate_on_first_solution: If True, stop as soon as any meeting is found.

    Returns:
        A JIT-compiled function that performs bidirectional Q* search
    """
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

    import warnings

    backward_mode = backward_mode.strip().lower()
    valid_backward_modes = {"auto", "edge_q", "value_v", "dijkstra"}
    if backward_mode not in valid_backward_modes:
        raise ValueError(
            f"Invalid backward_mode={backward_mode!r}. Expected one of {sorted(valid_backward_modes)}"
        )

    has_inverse_action_map = hasattr(puzzle, "inverse_action_map")
    if backward_mode == "edge_q" and (not has_inverse_action_map):
        warnings.warn(
            "bi_qstar backward_mode='edge_q' requires puzzle.inverse_action_map; "
            "falling back to backward_mode='value_v'.",
            UserWarning,
        )
        backward_mode = "value_v"

    if (not terminate_on_first_solution) and backward_mode != "dijkstra":
        warnings.warn(
            "bi_qstar with terminate_on_first_solution=False requires an admissible lower bound "
            "consistent with PQ keys. This is generally NOT guaranteed for learned/approximate Q. "
            "Use with care or prefer terminate_on_first_solution=True.",
            UserWarning,
        )

    use_backward_q = (not q_fn.is_fixed) and backward_mode != "dijkstra"
    init_loop_state, loop_condition, loop_body = _bi_qstar_loop_builder(
        puzzle,
        q_fn,
        bi_result_template,
        batch_size,
        cost_weight,
        look_ahead_pruning,
        pessimistic_update,
        use_backward_q=use_backward_q,
        backward_mode=backward_mode,
        terminate_on_first_solution=terminate_on_first_solution,
    )

    def bi_qstar(
        solve_config: Puzzle.SolveConfig,
        start: Puzzle.State,
        **kwargs: Any,
    ) -> BiDirectionalSearchResult:
        """Perform bidirectional Q* search."""
        # Prepare Q-function parameters for both directions
        q_params_forward = q_fn.prepare_q_parameters(solve_config, **kwargs)
        # Build a backward solve config that treats `start` as the target.
        # Prefer puzzle-level normalization via hindsight_transform.
        inverse_solveconfig = puzzle.hindsight_transform(solve_config, start)

        if use_backward_q:
            q_params_backward = q_fn.prepare_q_parameters(inverse_solveconfig, **kwargs)
        else:
            q_params_backward = q_params_forward

        loop_state = init_loop_state(
            bi_result_template,
            solve_config,
            inverse_solveconfig,
            start,
            q_params_forward,
            q_params_backward,
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

    return compile_search_builder(bi_qstar, puzzle, show_compile_time, warmup_inputs)
