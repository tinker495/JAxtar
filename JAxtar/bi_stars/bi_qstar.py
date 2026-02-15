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
import xtructure.numpy as xnp
from puxle import Puzzle

from helpers.jax_compile import jit_with_warmup
from JAxtar.annotate import KEY_DTYPE, MIN_BATCH_SIZE
from JAxtar.bi_stars.bi_search_base import (
    BiDirectionalSearchResult,
    BiLoopStateWithStates,
    build_bi_search_result,
    check_intersection,
    common_bi_loop_condition,
    finalize_bidirectional_result,
    initialize_bi_loop_common,
    materialize_meeting_point_hashidxs,
    update_meeting_point,
    update_meeting_point_best_only_deferred,
)
from JAxtar.stars.search_base import (
    Current,
    Parant_with_Costs,
    Parent,
    build_action_major_parent_context,
    insert_priority_queue_batches,
    packed_masked_state_eval,
    sort_and_pack_action_candidates,
)
from JAxtar.utils.batch_switcher import variable_batch_switcher_builder
from qfunction.q_base import QFunction

Q_EVAL_CHUNK_SIZE = 2048


def _bi_qstar_loop_builder(
    puzzle: Puzzle,
    q_fn: QFunction,
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
        batch_size: Batch size for parallel processing
        cost_weight: Weight for path cost in f = cost_weight * g + Q(s,a)
        look_ahead_pruning: Enable look-ahead pruning optimization
        pessimistic_update: Use max Q-value for duplicates (True) or min (False)

    Returns:
        Tuple of (init_loop_state, loop_condition, loop_body) functions
    """
    action_size = puzzle.action_size

    dist_sign = -1.0 if pessimistic_update else 1.0

    q_eval_chunk_size = min(batch_size, Q_EVAL_CHUNK_SIZE)
    q_eval_min_batch = min(MIN_BATCH_SIZE, q_eval_chunk_size)
    variable_q_batch_switcher = variable_batch_switcher_builder(
        q_fn.batched_q_value,
        max_batch_size=q_eval_chunk_size,
        min_batch_size=q_eval_min_batch,
        pad_value=jnp.inf,
    )
    q_eval_num_chunks = (batch_size + q_eval_chunk_size - 1) // q_eval_chunk_size
    q_eval_padded_size = q_eval_num_chunks * q_eval_chunk_size

    def _batched_q_eval(q_params: Any, states: Puzzle.State, filled: chex.Array) -> chex.Array:
        """Evaluate Q-values in fixed chunks to limit peak memory."""
        if q_eval_chunk_size == batch_size:
            return variable_q_batch_switcher(q_params, states, filled)

        pad_size = q_eval_padded_size - batch_size
        padded_states = xnp.pad(states, (0, pad_size))
        padded_filled = jnp.pad(filled, (0, pad_size), constant_values=False)
        states_chunked = padded_states.reshape((q_eval_num_chunks, q_eval_chunk_size))
        filled_chunked = padded_filled.reshape((q_eval_num_chunks, q_eval_chunk_size))

        def _scan(_, inputs):
            states_slice, compute_mask = inputs
            q_vals = variable_q_batch_switcher(q_params, states_slice, compute_mask)
            return None, q_vals

        _, q_chunks = jax.lax.scan(_scan, None, (states_chunked, filled_chunked))
        q_vals = q_chunks.reshape((q_eval_padded_size, action_size))
        return q_vals[:batch_size]

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

    def _expand_direction_q(
        bi_result: BiDirectionalSearchResult,
        solve_config: Puzzle.SolveConfig,
        inverse_solveconfig: Puzzle.SolveConfig,
        q_params: Any,
        current: Current,
        states: Puzzle.State,
        filled: chex.Array,
        is_forward: bool,
        use_q: bool,
    ) -> tuple[BiDirectionalSearchResult, Current, Puzzle.State, chex.Array]:
        """
        Expand one direction using Q-function evaluation.

        Q* evaluates Q(s, a) on parent states to get action-dependent values.
        """
        if is_forward:
            search_result = bi_result.forward
            opposite_sr = bi_result.backward
            current_solve_config = solve_config
            get_neighbours_fn = puzzle.batched_get_neighbours
        else:
            search_result = bi_result.backward
            opposite_sr = bi_result.forward
            current_solve_config = inverse_solveconfig
            get_neighbours_fn = puzzle.batched_get_inverse_neighbours

        sr_batch_size = search_result.batch_size

        cost = current.cost
        hash_idx = current.hashidx

        (
            flat_parent_hashidx,
            flat_actions,
            costs,
            filled_tiles,
            unflatten_shape,
        ) = build_action_major_parent_context(
            hash_idx,
            cost,
            filled,
            action_size,
            sr_batch_size,
        )

        # For backward direction, action indices refer to inverse neighbours.
        # Action-dependent Q(s, a) on the backward "parent" state is not semantically aligned.
        # However, if inverse_action_map is available, we can map inverse action indices
        # to the corresponding forward actions effectively allowing us to use Q(s, a)
        # as a heuristic for the edge (s, s').
        # Backward Q semantics are domain-dependent.
        # - edge_q: use Q(parent, a) for backward edges by remapping actions via inverse_action_map.
        # - value_v: use V(s)=min_a Q(s,a) as a heuristic on predecessor states (q-guided A*).
        # - auto: edge_q if inverse_action_map exists, else value_v.
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
                # backward_mode == "dijkstra" should be implemented by calling this function
                # with use_q=False for the backward direction.
                can_optimize_bwd = False
                use_value_heuristic = False
        else:
            can_optimize_bwd = False
            use_value_heuristic = False
        use_heuristic_in_pop = use_value_heuristic

        # We need neighbor generation when:
        # - look-ahead pruning is enabled, or
        # - backward uses value-heuristic (V(child)), or
        # - Q is disabled (we need step costs to score edges / build a Dijkstra fallback).
        need_neighbours = look_ahead_pruning or use_value_heuristic or (not use_q)
        if need_neighbours:
            neighbour_look_a_head, ncosts = get_neighbours_fn(current_solve_config, states, filled)
            look_a_head_costs = (costs + ncosts).astype(KEY_DTYPE)

        flattened_filled_tiles = filled_tiles.flatten()

        if use_value_heuristic:
            # Heuristic on predecessor states: v(s) = min_a Q(s, a)
            flattened_neighbour_look_head = neighbour_look_a_head.flatten()
            flattened_look_a_head_costs = look_a_head_costs.flatten().astype(KEY_DTYPE)

            # Compute V(child)=min_a Q(child,a) in a packed+chunked way to avoid
            # calling the Q-function with leading dim = action_size * batch_size.
            flat_states = flattened_neighbour_look_head
            flat_mask = flattened_filled_tiles
            heuristic_vals = packed_masked_state_eval(
                flat_states,
                flat_mask,
                action_size,
                sr_batch_size,
                lambda states_slice, compute_mask: jnp.min(
                    _batched_q_eval(q_params, states_slice, compute_mask),
                    axis=-1,
                ),
                dtype=KEY_DTYPE,
            )
            heuristic_vals = jnp.where(filled_tiles, heuristic_vals, jnp.inf).astype(KEY_DTYPE)

            neighbour_keys = (cost_weight * look_a_head_costs + heuristic_vals).astype(KEY_DTYPE)
            neighbour_keys = jnp.where(filled_tiles, neighbour_keys, jnp.inf)

            flattened_keys = neighbour_keys.flatten()
            dists = heuristic_vals.flatten()

            # Meeting mask for value heuristic branch
            # Relaxed mask for meeting usage: unique state with best cost in this batch.
            meeting_mask = (
                xnp.unique_mask(
                    flattened_neighbour_look_head,
                    flattened_look_a_head_costs,
                    flattened_filled_tiles,
                )
                & flattened_filled_tiles
            )

            if look_ahead_pruning:
                current_hash_idxs, found = search_result.hashtable.lookup_parallel(
                    flattened_neighbour_look_head, meeting_mask
                )
                old_costs = search_result.get_cost(current_hash_idxs)
                candidate_mask = meeting_mask & jnp.logical_or(
                    ~found, jnp.less(flattened_look_a_head_costs, old_costs)
                )

                # `meeting_mask` is already unique-by-cost, so `candidate_mask` is unique as well.
                optimal_mask = candidate_mask
            else:
                optimal_mask = flattened_filled_tiles

            # Best-only early meeting update without HT insertion.
            # Reuse this-direction lookup results when available.
            if look_ahead_pruning:
                this_hashidx_all, this_found_all, this_old_costs_all = (
                    current_hash_idxs,
                    found,
                    old_costs,
                )
            else:
                this_hashidx_all, this_found_all = search_result.hashtable.lookup_parallel(
                    flattened_neighbour_look_head, meeting_mask
                )
                this_old_costs_all = search_result.get_cost(this_hashidx_all)
            bi_result.meeting = update_meeting_point_best_only_deferred(
                bi_result.meeting,
                this_sr=search_result,
                opposite_sr=opposite_sr,
                candidate_states=flattened_neighbour_look_head,
                candidate_costs=flattened_look_a_head_costs,
                candidate_mask=meeting_mask,  # Use relaxed mask
                this_found=this_found_all,
                this_hashidx=this_hashidx_all,
                this_old_costs=this_old_costs_all,
                this_parent_hashidx=flat_parent_hashidx,
                this_parent_action=flat_actions,
                is_forward=is_forward,
            )

        else:
            if use_q:
                # Forward direction: Q(s, a) on parent states.
                q_vals = _batched_q_eval(q_params, states, filled)
                q_vals = q_vals.transpose().astype(KEY_DTYPE)  # [action_size, batch_size]

                if can_optimize_bwd:
                    inv_map = puzzle.inverse_action_map
                    q_vals = q_vals[inv_map, :]
            else:
                # Fallback: use true step costs so pop maps dist -> 0 via (step_cost - step_cost).
                q_vals = ncosts.astype(KEY_DTYPE)

            neighbour_keys = (cost_weight * costs + q_vals).astype(KEY_DTYPE)
            neighbour_keys = jnp.where(filled_tiles, neighbour_keys, jnp.inf)

            flattened_keys = neighbour_keys.flatten()
            raw_q_flat = q_vals.flatten()
            dists = raw_q_flat

            if look_ahead_pruning:
                flattened_neighbour_look_head = neighbour_look_a_head.flatten()
                flattened_look_a_head_costs = look_a_head_costs.flatten().astype(KEY_DTYPE)

                distinct_score = flattened_look_a_head_costs + dist_sign * 1e-5 * dists

                unique_mask = xnp.unique_mask(
                    flattened_neighbour_look_head,
                    distinct_score,
                    flattened_filled_tiles,
                )
                current_hash_idxs, found = search_result.hashtable.lookup_parallel(
                    flattened_neighbour_look_head, unique_mask
                )
                old_costs = search_result.get_cost(current_hash_idxs)
                old_dists = search_result.get_dist(current_hash_idxs)

                if use_q:
                    # `old_dists` is stored for the *state* after pop.
                    # In deferred Q*, pop stores: dist(state) = Q(parent, action) - step_cost.
                    # To compare/update in Q-space we reconstruct:
                    #   Q_old(parent->child) = old_dist(child) + step_cost(parent->child)
                    q_old_reconstructed = old_dists.astype(KEY_DTYPE) + ncosts.flatten().astype(
                        KEY_DTYPE
                    )
                    if pessimistic_update:
                        # Max over Q-values
                        q_old_for_max = jnp.where(found, q_old_reconstructed, -jnp.inf)
                        dists = jnp.maximum(dists, q_old_for_max)
                    else:
                        # Min over Q-values
                        q_old_for_min = jnp.where(found, q_old_reconstructed, jnp.inf)
                        dists = jnp.minimum(dists, q_old_for_min)

                better_cost_mask = jnp.less(flattened_look_a_head_costs, old_costs)
                optimal_mask = unique_mask & (jnp.logical_or(~found, better_cost_mask))

                # Meeting mask for regular Q branch with look ahead
                # Relaxed mask for meeting usage: unique state with best cost in this batch.
                meeting_mask = (
                    xnp.unique_mask(
                        flattened_neighbour_look_head,
                        flattened_look_a_head_costs,
                        flattened_filled_tiles,
                    )
                    & flattened_filled_tiles
                )

                # Best-only early meeting update without HT insertion.
                this_hashidx_all, this_found_all = search_result.hashtable.lookup_parallel(
                    flattened_neighbour_look_head, meeting_mask
                )
                this_old_costs_all = search_result.get_cost(this_hashidx_all)
                bi_result.meeting = update_meeting_point_best_only_deferred(
                    bi_result.meeting,
                    this_sr=search_result,
                    opposite_sr=opposite_sr,
                    candidate_states=flattened_neighbour_look_head,
                    candidate_costs=flattened_look_a_head_costs,
                    candidate_mask=meeting_mask,  # Use relaxed mask
                    this_found=this_found_all,
                    this_hashidx=this_hashidx_all,
                    this_old_costs=this_old_costs_all,
                    this_parent_hashidx=flat_parent_hashidx,
                    this_parent_action=flat_actions,
                    is_forward=is_forward,
                )
            else:
                optimal_mask = flattened_filled_tiles

        # Create values for priority queue
        flattened_vals = Parant_with_Costs(
            parent=Parent(hashidx=flat_parent_hashidx, action=flat_actions),
            cost=costs.flatten(),
            dist=dists,
        )

        (
            neighbour_keys_reshaped,
            vals_reshaped,
            optimal_mask_reshaped,
        ) = sort_and_pack_action_candidates(
            flattened_keys,
            flattened_vals,
            optimal_mask,
            action_size,
            sr_batch_size,
        )

        search_result = insert_priority_queue_batches(
            search_result,
            neighbour_keys_reshaped,
            vals_reshaped,
            optimal_mask_reshaped,
        )

        # Pop next batch with states
        search_result, new_current, new_states, new_filled = search_result.pop_full_with_actions(
            puzzle=puzzle,
            solve_config=solve_config,
            use_heuristic=use_heuristic_in_pop,
            is_backward=not is_forward,
        )

        # Update bi_result
        if is_forward:
            bi_result.forward = search_result
        else:
            bi_result.backward = search_result

        # Check if newly popped states exist in opposite HT
        # At this point, new_current.hashidx is valid (states are inserted during pop)
        (
            new_found_mask,
            new_opposite_hashidx,
            new_opposite_costs,
            new_total_costs,
        ) = check_intersection(
            new_states,
            new_current.cost,
            new_filled,
            opposite_sr,
        )

        bi_result.meeting = update_meeting_point(
            bi_result.meeting,
            new_found_mask,
            new_current.hashidx,
            new_opposite_hashidx,
            new_current.cost,
            new_opposite_costs,
            new_total_costs,
            is_forward,
        )

        return bi_result, new_current, new_states, new_filled

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
                is_forward=True,
                use_q=True,
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
                is_forward=False,
                use_q=use_backward_q,
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

        # Build per-call search storage inside the jitted function to avoid
        # capturing large templates as compile-time constants.
        bi_result = build_bi_search_result(
            statecls,
            batch_size,
            max_nodes,
            action_size,
            pop_ratio=pop_ratio,
            min_pop=min_pop,
            parant_with_costs=True,
            # Q* keeps larger deferred queues; a tighter hash-table multiplier
            # lowers peak memory without changing the per-direction max_nodes budget.
            hash_size_multiplier=1,
        )

        loop_state = init_loop_state(
            bi_result,
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

        return finalize_bidirectional_result(bi_result)

    return jit_with_warmup(
        bi_qstar,
        puzzle=puzzle,
        show_compile_time=show_compile_time,
        warmup_inputs=warmup_inputs,
        init_message="Initializing JIT for bidirectional Q*...",
        completion_message="JIT compiled\n",
    )
