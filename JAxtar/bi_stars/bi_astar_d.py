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

import time
from typing import Any

import chex
import jax
import jax.numpy as jnp
import xtructure.numpy as xnp
from puxle import Puzzle

from helpers.jax_compile import compile_with_example
from heuristic.heuristic_base import Heuristic
from JAxtar.annotate import ACTION_DTYPE, KEY_DTYPE, MIN_BATCH_UNIT
from JAxtar.bi_stars.bi_search_base import (
    BiDirectionalSearchResult,
    BiLoopStateWithStates,
    build_bi_search_result,
    check_intersection,
    common_bi_loop_condition,
    initialize_bi_loop_common,
    materialize_meeting_point_hashidxs,
    update_meeting_point,
    update_meeting_point_best_only_deferred,
)
from JAxtar.stars.search_base import Current, Parant_with_Costs, Parent, SearchResult
from JAxtar.utils.array_ops import stable_partition_three
from JAxtar.utils.batch_switcher import (
    build_batch_sizes_for_cap,
    variable_batch_switcher_builder,
)


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
    heuristic_batch_sizes = build_batch_sizes_for_cap(batch_size, min_batch_unit=MIN_BATCH_UNIT)

    variable_heuristic_batch_switcher = variable_batch_switcher_builder(
        heuristic.batched_distance,
        pad_value=jnp.inf,
        max_batch_size=batch_size,
        batch_sizes=heuristic_batch_sizes,
        partition_mode="flat",
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

    def _expand_direction_deferred(
        bi_result: BiDirectionalSearchResult,
        solve_config: Puzzle.SolveConfig,
        inverse_solveconfig: Puzzle.SolveConfig,
        heuristic_params: Any,
        current: Current,
        states: Puzzle.State,
        filled: chex.Array,
        is_forward: bool,
        use_heuristic: bool,
    ) -> tuple[BiDirectionalSearchResult, Current, Puzzle.State, chex.Array]:
        """
        Expand one direction using deferred heuristic evaluation.

        In deferred A*, we insert (parent, action) pairs into the PQ using
        the parent's f-value. Heuristics are computed when nodes are popped.
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
        flat_size = action_size * sr_batch_size

        cost = current.cost
        hash_idx = current.hashidx

        idx_tiles = xnp.tile(hash_idx, (action_size, 1))
        action = jnp.tile(
            jnp.arange(action_size, dtype=ACTION_DTYPE)[:, jnp.newaxis],
            (1, sr_batch_size),
        )
        costs = jnp.tile(cost[jnp.newaxis, :], (action_size, 1))
        filled_tiles = jnp.tile(filled[jnp.newaxis, :], (action_size, 1))

        flattened_filled_tiles = filled_tiles.flatten()

        if look_ahead_pruning:
            # Look-ahead: compute neighbors and filter before inserting into PQ
            neighbour_look_a_head, ncosts = get_neighbours_fn(current_solve_config, states, filled)
            look_a_head_costs = (costs + ncosts).astype(KEY_DTYPE)

            flattened_neighbour_look_head = neighbour_look_a_head.flatten()
            flattened_look_a_head_costs = look_a_head_costs.flatten().astype(KEY_DTYPE)

            # Meeting update mask: unique (min cost) among current batch.
            # We compute this first so we can restrict hash-table lookups to one
            # representative per state.
            meeting_mask = (
                xnp.unique_mask(
                    flattened_neighbour_look_head,
                    flattened_look_a_head_costs,
                    flattened_filled_tiles,
                )
                & flattened_filled_tiles
            )

            current_hash_idxs, found = search_result.hashtable.lookup_parallel(
                flattened_neighbour_look_head, meeting_mask
            )

            old_costs = search_result.get_cost(current_hash_idxs)

            # PQ insertion mask: Must improve cost or be new in THIS direction's HT
            candidate_mask = meeting_mask & jnp.logical_or(
                ~found, jnp.less(flattened_look_a_head_costs, old_costs)
            )

            # `meeting_mask` is already unique-by-cost, so `candidate_mask` is unique as well.
            optimal_mask = candidate_mask

            if use_heuristic:
                found_reshaped = found.reshape(action_size, sr_batch_size)
                optimal_mask_reshaped = optimal_mask.reshape(action_size, sr_batch_size)

                old_dists = search_result.get_dist(current_hash_idxs)
                old_dists = old_dists.reshape(action_size, sr_batch_size)

                need_compute = optimal_mask_reshaped & ~found_reshaped

                # Pack contiguously for efficient heuristic computation
                flat_states = neighbour_look_a_head.flatten()
                flat_need_compute = need_compute.flatten()

                n = flat_size
                n = flat_size
                sorted_indices = stable_partition_three(
                    flat_need_compute, jnp.zeros_like(flat_need_compute, dtype=jnp.bool_)
                )

                sorted_states = flat_states[sorted_indices]
                sorted_mask = flat_need_compute[sorted_indices]

                # Automated batch processing
                h_val_sorted = variable_heuristic_batch_switcher(
                    heuristic_params, sorted_states, sorted_mask
                )
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

            # Optional backward value-lookahead: use a 1-step backup on predecessor states
            # to better match Q*(s,a)=step+h(next) behavior in the backward direction.
            # For RubiksCubeQ, this corresponds to V(s)=min_a (1 + h(s')) which is strictly
            # more informative than h(s) and can dramatically reduce expansions.
            if (not is_forward) and use_heuristic and backward_value_lookahead:
                # Use the already computed key (g + h) to pick a small set of best candidates
                # and run the expensive backup only on those.
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
                # `variable_heuristic_batch_switcher` is built with max_batch_size=batch_size,
                # so keep the backup batch <= sr_batch_size.
                k = min(k, sr_batch_size, flat_size)
                sel_idx = sorted_idx[:k]
                sel_valid = jnp.isfinite(sorted_keys[:k])

                sel_states = flattened_neighbour_look_head[sel_idx]
                succ_states, succ_costs = puzzle.batched_get_neighbours(
                    solve_config, sel_states, sel_valid
                )  # [action, k], [action, k]

                tiled_valid = jnp.tile(sel_valid[jnp.newaxis, :], (action_size, 1))
                flat_valid = tiled_valid.flatten()
                flat_succ_states = succ_states.flatten()
                flat_succ_costs = succ_costs.flatten()

                # Pack valid entries to front for correct batch_switcher behavior
                pack_perm = stable_partition_three(
                    flat_valid, jnp.zeros_like(flat_valid, dtype=jnp.bool_)
                )
                packed_states = flat_succ_states[pack_perm]
                packed_valid = flat_valid[pack_perm]

                packed_h_succ = variable_heuristic_batch_switcher(
                    heuristic_params, packed_states, packed_valid
                ).astype(KEY_DTYPE)

                # Unpack results back to original order
                flat_h_succ = jnp.empty_like(packed_h_succ).at[pack_perm].set(packed_h_succ)

                q_chunks_flat = flat_h_succ + flat_succ_costs.astype(KEY_DTYPE)
                q_chunks = q_chunks_flat.reshape(action_size, k)
                v_sel = jnp.min(q_chunks, axis=0).astype(KEY_DTYPE)

                heur_flat = heuristic_vals.flatten()
                updated = jnp.where(sel_valid, v_sel, heur_flat[sel_idx])
                heur_flat = heur_flat.at[sel_idx].set(updated)
                heuristic_vals = heur_flat.reshape((action_size, sr_batch_size))

            neighbour_keys = (cost_weight * look_a_head_costs + heuristic_vals).astype(KEY_DTYPE)

            # Early meeting detection for deferred variants (0 HT insert):
            # Update the meeting upper bound without inserting the meeting state.
            # If the meeting state doesn't exist in this direction's HT yet, store
            # its last-edge (parent, action) so it can be materialized later.
            bi_result.meeting = update_meeting_point_best_only_deferred(
                bi_result.meeting,
                this_sr=search_result,
                opposite_sr=opposite_sr,
                candidate_states=flattened_neighbour_look_head,
                candidate_costs=flattened_look_a_head_costs,
                candidate_mask=meeting_mask,
                this_found=found,
                this_hashidx=current_hash_idxs,
                this_old_costs=old_costs,
                this_parent_hashidx=idx_tiles.flatten(),
                this_parent_action=action.flatten(),
                is_forward=is_forward,
            )

        else:
            # No look-ahead: use parent's heuristic value
            if use_heuristic:
                heuristic_vals = variable_heuristic_batch_switcher(heuristic_params, states, filled)
                heuristic_vals = jnp.where(filled, heuristic_vals, jnp.inf)
                heuristic_vals = jnp.tile(heuristic_vals[jnp.newaxis, :], (action_size, 1)).astype(
                    KEY_DTYPE
                )
            else:
                heuristic_vals = jnp.zeros_like(costs, dtype=KEY_DTYPE)
                heuristic_vals = jnp.where(filled_tiles, heuristic_vals, jnp.inf)

            optimal_mask = flattened_filled_tiles
            neighbour_keys = (cost_weight * costs + heuristic_vals).astype(KEY_DTYPE)

        # Create values for priority queue
        vals = Parant_with_Costs(
            parent=Parent(hashidx=idx_tiles.flatten(), action=action.flatten()),
            cost=costs.flatten(),
            dist=heuristic_vals.flatten(),
        )
        flattened_vals = vals.flatten()
        flattened_keys = neighbour_keys.flatten()

        flattened_neighbour_keys = jnp.where(optimal_mask, flattened_keys, jnp.inf)

        # Sort for efficiency
        sorted_key, sorted_idx = jax.lax.sort_key_val(
            flattened_neighbour_keys, jnp.arange(flat_size)
        )
        sorted_vals = flattened_vals[sorted_idx]
        sorted_optimal_mask = optimal_mask[sorted_idx]

        neighbour_keys_reshaped = sorted_key.reshape(action_size, sr_batch_size)
        vals_reshaped = sorted_vals.reshape((action_size, sr_batch_size))
        optimal_mask_reshaped = sorted_optimal_mask.reshape(action_size, sr_batch_size)

        def _insert(sr: SearchResult, keys, vals):
            sr.priority_queue = sr.priority_queue.insert(keys, vals)
            return sr

        def _scan(sr: SearchResult, val):
            keys, vals, mask = val
            sr = jax.lax.cond(
                jnp.any(mask),
                _insert,
                lambda sr, *args: sr,
                sr,
                keys,
                vals,
            )
            return sr, None

        search_result, _ = jax.lax.scan(
            _scan,
            search_result,
            (neighbour_keys_reshaped, vals_reshaped, optimal_mask_reshaped),
        )

        # Pop next batch with states
        search_result, new_current, new_states, new_filled = search_result.pop_full_with_actions(
            puzzle=puzzle,
            solve_config=solve_config,
            use_heuristic=True,
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
                is_forward=True,
                use_heuristic=True,
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
                is_forward=False,
                use_heuristic=use_backward_heuristic,
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
    min_pop = max(1, MIN_BATCH_UNIT // denom)

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

    bi_astar_d_fn = jax.jit(bi_astar_d)
    if show_compile_time:
        print("Initializing JIT for bidirectional A* deferred...")
        start_time = time.time()

    if warmup_inputs is None:
        empty_solve_config = puzzle.SolveConfig.default()
        empty_states = puzzle.State.default()
        bi_astar_d_fn(empty_solve_config, empty_states)
    else:
        compile_with_example(bi_astar_d_fn, *warmup_inputs)

    if show_compile_time:
        end_time = time.time()
        compile_time = end_time - start_time
        print("Compile Time: {:6.2f} seconds".format(compile_time))
        print("JIT compiled\n")

    return bi_astar_d_fn
