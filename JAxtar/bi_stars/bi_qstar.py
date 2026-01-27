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

import time
from typing import Any

import chex
import jax
import jax.numpy as jnp
import xtructure.numpy as xnp
from puxle import Puzzle

from JAxtar.annotate import ACTION_DTYPE, KEY_DTYPE, MIN_BATCH_SIZE
from JAxtar.bi_stars.bi_search_base import (
    BiDirectionalSearchResult,
    BiLoopStateWithStates,
    MeetingPoint,
    bi_termination_condition,
    build_bi_search_result,
    check_intersection,
    get_min_f_value,
    materialize_meeting_point_hashidxs,
    update_meeting_point,
    update_meeting_point_best_only_deferred,
)
from JAxtar.stars.search_base import Current, Parant_with_Costs, Parent, SearchResult
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

    dist_sign = -1.0 if pessimistic_update else 1.0

    variable_q_batch_switcher = variable_batch_switcher_builder(
        q_fn.batched_q_value,
        max_batch_size=batch_size,
        min_batch_size=MIN_BATCH_SIZE,
        pad_value=jnp.inf,
    )

    def init_loop_state(
        bi_result: BiDirectionalSearchResult,
        solve_config: Puzzle.SolveConfig,
        start: Puzzle.State,
        q_params_forward: Any,
        q_params_backward: Any,
    ) -> BiLoopStateWithStates:
        """Initialize bidirectional Q* search from start and goal states."""

        sr_batch_size = bi_result.batch_size

        # Initialize forward search (from start)
        bi_result.forward.hashtable, _, fwd_hash_idx = bi_result.forward.hashtable.insert(start)
        bi_result.forward.cost = bi_result.forward.cost.at[fwd_hash_idx.index].set(0)

        fwd_hash_idxs = xnp.pad(fwd_hash_idx, (0, sr_batch_size - 1))
        fwd_costs = jnp.zeros((sr_batch_size,), dtype=KEY_DTYPE)
        fwd_states = xnp.pad(start, (0, sr_batch_size - 1))
        fwd_filled = jnp.zeros(sr_batch_size, dtype=jnp.bool_).at[0].set(True)

        # Initialize backward search (from goal)
        # Use puzzle-level transform to obtain a concrete goal state.
        goal = puzzle.solve_config_to_state_transform(solve_config, key=jax.random.PRNGKey(0))
        bi_result.backward.hashtable, _, bwd_hash_idx = bi_result.backward.hashtable.insert(goal)
        bi_result.backward.cost = bi_result.backward.cost.at[bwd_hash_idx.index].set(0)

        bwd_hash_idxs = xnp.pad(bwd_hash_idx, (0, sr_batch_size - 1))
        bwd_costs = jnp.zeros((sr_batch_size,), dtype=KEY_DTYPE)
        bwd_states = xnp.pad(goal, (0, sr_batch_size - 1))
        bwd_filled = jnp.zeros(sr_batch_size, dtype=jnp.bool_).at[0].set(True)

        # Check if start == goal (cost = 0 case)
        start_in_bwd_idx, start_in_bwd_found = bi_result.backward.hashtable.lookup(start)
        is_same = jnp.logical_and(start_in_bwd_found, start_in_bwd_idx.index == bwd_hash_idx.index)

        dummy_hashidx = fwd_hash_idx
        dummy_action = jnp.array(0, dtype=ACTION_DTYPE)

        bi_result.meeting = jax.lax.cond(
            is_same,
            lambda _: MeetingPoint(
                fwd_hashidx=fwd_hash_idx,
                bwd_hashidx=bwd_hash_idx,
                fwd_cost=jnp.array(0.0, dtype=KEY_DTYPE),
                bwd_cost=jnp.array(0.0, dtype=KEY_DTYPE),
                total_cost=jnp.array(0.0, dtype=KEY_DTYPE),
                found=jnp.array(True),
                fwd_has_hashidx=jnp.array(True),
                bwd_has_hashidx=jnp.array(True),
                fwd_parent_hashidx=dummy_hashidx,
                fwd_parent_action=dummy_action,
                bwd_parent_hashidx=dummy_hashidx,
                bwd_parent_action=dummy_action,
            ),
            lambda _: bi_result.meeting,
            None,
        )

        return BiLoopStateWithStates(
            bi_result=bi_result,
            solve_config=solve_config,
            params_forward=q_params_forward,
            params_backward=q_params_backward,
            current_forward=Current(hashidx=fwd_hash_idxs, cost=fwd_costs),
            current_backward=Current(hashidx=bwd_hash_idxs, cost=bwd_costs),
            states_forward=fwd_states,
            states_backward=bwd_states,
            filled_forward=fwd_filled,
            filled_backward=bwd_filled,
        )

    def loop_condition(loop_state: BiLoopStateWithStates) -> chex.Array:
        """Check if search should continue."""
        bi_result = loop_state.bi_result

        # Check if queues have nodes
        fwd_has_nodes = loop_state.filled_forward.any()
        bwd_has_nodes = loop_state.filled_backward.any()

        # Check hash table capacity per direction.
        # If one direction is full, we can still expand the other direction and
        # potentially intersect with the already-built frontier.
        fwd_not_full = bi_result.forward.generated_size < bi_result.forward.capacity
        bwd_not_full = bi_result.backward.generated_size < bi_result.backward.capacity
        has_work = jnp.logical_or(
            jnp.logical_and(fwd_has_nodes, fwd_not_full),
            jnp.logical_and(bwd_has_nodes, bwd_not_full),
        )

        # Check termination condition
        fwd_min_f = get_min_f_value(
            bi_result.forward, loop_state.current_forward, loop_state.filled_forward, cost_weight
        )
        bwd_min_f = get_min_f_value(
            bi_result.backward, loop_state.current_backward, loop_state.filled_backward, cost_weight
        )

        should_terminate = bi_termination_condition(bi_result, fwd_min_f, bwd_min_f, cost_weight)

        return jnp.logical_and(has_work, ~should_terminate)

    def _expand_direction_q(
        bi_result: BiDirectionalSearchResult,
        solve_config: Puzzle.SolveConfig,
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
            get_neighbours_fn = puzzle.batched_get_neighbours
        else:
            search_result = bi_result.backward
            opposite_sr = bi_result.forward
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

        # For backward direction, action indices refer to inverse neighbours.
        # Action-dependent Q(s, a) on the backward "parent" state is not semantically aligned.
        # Instead, when Q is available (retargetable), we derive a state-value heuristic on the
        # predecessor states and run a deferred A*-style ordering in the backward direction.
        use_value_heuristic = (not is_forward) and use_q
        use_heuristic_in_pop = use_value_heuristic

        need_neighbours = look_ahead_pruning or (not use_q) or use_value_heuristic
        if need_neighbours:
            neighbour_look_a_head, ncosts = get_neighbours_fn(solve_config, states, filled)
            look_a_head_costs = (costs + ncosts).astype(KEY_DTYPE)

        flattened_filled_tiles = filled_tiles.flatten()

        if use_value_heuristic:
            # Heuristic on predecessor states: v(s) = min_a Q(s, a)
            flattened_neighbour_look_head = neighbour_look_a_head.flatten()
            flattened_look_a_head_costs = look_a_head_costs.flatten().astype(KEY_DTYPE)

            q_child = q_fn.batched_q_value(
                q_params, flattened_neighbour_look_head
            )  # [flat, action]
            v_child = jnp.min(q_child, axis=-1).astype(KEY_DTYPE)  # [flat]
            heuristic_vals = v_child.reshape(action_size, sr_batch_size)
            heuristic_vals = jnp.where(filled_tiles, heuristic_vals, jnp.inf).astype(KEY_DTYPE)

            neighbour_keys = (cost_weight * look_a_head_costs + heuristic_vals).astype(KEY_DTYPE)
            neighbour_keys = jnp.where(filled_tiles, neighbour_keys, jnp.inf)

            flattened_keys = neighbour_keys.flatten()
            dists = heuristic_vals.flatten()

            if look_ahead_pruning:
                current_hash_idxs, found = search_result.hashtable.lookup_parallel(
                    flattened_neighbour_look_head, flattened_filled_tiles
                )
                old_costs = search_result.get_cost(current_hash_idxs)
                candidate_mask = jnp.logical_or(
                    ~found, jnp.less(flattened_look_a_head_costs, old_costs)
                )
                candidate_mask = candidate_mask & flattened_filled_tiles
                unique_mask = xnp.unique_mask(
                    flattened_neighbour_look_head,
                    flattened_look_a_head_costs,
                    candidate_mask,
                )
                optimal_mask = unique_mask & candidate_mask
            else:
                optimal_mask = flattened_filled_tiles

            # Best-only early meeting update without HT insertion.
            # Ensure we have this-direction lookup info for meeting accounting.
            this_hashidx_all, this_found_all = search_result.hashtable.lookup_parallel(
                flattened_neighbour_look_head, flattened_filled_tiles
            )
            this_old_costs_all = search_result.get_cost(this_hashidx_all)
            bi_result.meeting = update_meeting_point_best_only_deferred(
                bi_result.meeting,
                this_sr=search_result,
                opposite_sr=opposite_sr,
                candidate_states=flattened_neighbour_look_head,
                candidate_costs=flattened_look_a_head_costs,
                candidate_mask=optimal_mask,
                this_found=this_found_all,
                this_hashidx=this_hashidx_all,
                this_old_costs=this_old_costs_all,
                this_parent_hashidx=idx_tiles.flatten(),
                this_parent_action=action.flatten(),
                is_forward=is_forward,
            )

        else:
            if use_q:
                # Forward direction: Q(s, a) on parent states.
                q_vals = variable_q_batch_switcher(q_params, states, filled)
                q_vals = q_vals.transpose().astype(KEY_DTYPE)  # [action_size, batch_size]
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
                    if pessimistic_update:
                        safe_old_dists = jnp.where(found, old_dists, -jnp.inf)
                        dists = jnp.maximum(dists, safe_old_dists)
                    else:
                        safe_old_dists = jnp.where(found, old_dists, jnp.inf)
                        dists = jnp.minimum(dists, safe_old_dists)

                better_cost_mask = jnp.less(flattened_look_a_head_costs, old_costs)
                optimal_mask = unique_mask & (jnp.logical_or(~found, better_cost_mask))

                # Best-only early meeting update without HT insertion.
                this_hashidx_all, this_found_all = search_result.hashtable.lookup_parallel(
                    flattened_neighbour_look_head, flattened_filled_tiles
                )
                this_old_costs_all = search_result.get_cost(this_hashidx_all)
                bi_result.meeting = update_meeting_point_best_only_deferred(
                    bi_result.meeting,
                    this_sr=search_result,
                    opposite_sr=opposite_sr,
                    candidate_states=flattened_neighbour_look_head,
                    candidate_costs=flattened_look_a_head_costs,
                    candidate_mask=optimal_mask,
                    this_found=this_found_all,
                    this_hashidx=this_hashidx_all,
                    this_old_costs=this_old_costs_all,
                    this_parent_hashidx=idx_tiles.flatten(),
                    this_parent_action=action.flatten(),
                    is_forward=is_forward,
                )
            else:
                optimal_mask = flattened_filled_tiles

        # Create values for priority queue
        flattened_vals = Parant_with_Costs(
            parent=Parent(hashidx=idx_tiles.flatten(), action=action.flatten()),
            cost=costs.flatten(),
            dist=dists,
        )

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

        fwd_not_full = bi_result.forward.generated_size < bi_result.forward.capacity
        bwd_not_full = bi_result.backward.generated_size < bi_result.backward.capacity

        def _expand_forward(bi_result):
            return _expand_direction_q(
                bi_result,
                solve_config,
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

    use_backward_q = not q_fn.is_fixed
    init_loop_state, loop_condition, loop_body = _bi_qstar_loop_builder(
        puzzle,
        q_fn,
        bi_result_template,
        batch_size,
        cost_weight,
        look_ahead_pruning,
        pessimistic_update,
        use_backward_q=use_backward_q,
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
        if use_backward_q:
            backward_solve_config = puzzle.hindsight_transform(solve_config, start)
            q_params_backward = q_fn.prepare_q_parameters(backward_solve_config, **kwargs)
        else:
            q_params_backward = q_params_forward

        loop_state = init_loop_state(
            bi_result_template,
            solve_config,
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

    bi_qstar_fn = jax.jit(bi_qstar)
    empty_solve_config = puzzle.SolveConfig.default()
    empty_states = puzzle.State.default()

    if show_compile_time:
        print("Initializing JIT for bidirectional Q*...")
        start_time = time.time()

    bi_qstar_fn(empty_solve_config, empty_states)

    if show_compile_time:
        end_time = time.time()
        print(f"Compile Time: {end_time - start_time:6.2f} seconds")
        print("JIT compiled\n")

    return bi_qstar_fn
