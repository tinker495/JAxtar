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

from helpers.jax_compile import compile_search_builder
from JAxtar.annotate import KEY_DTYPE, MIN_BATCH_SIZE
from JAxtar.search_build_spec import (
    DEFAULT_SEARCH_BUILD_SPEC,
    SearchBuildSpec,
    _require_no_workload_signature,
)
from JAxtar.bi_stars.bi_search_base import (
    BiDirectionalSearchResult,
    BiLoopStateWithStates,
    build_bi_search_result,
    common_bi_loop_condition,
    initialize_bi_loop_common,
    materialize_meeting_point_hashidxs,
    stamp_bi_solved_from_meeting,
    build_bi_deferred_expand_direction,
)
from JAxtar.utils.batch_switcher import variable_batch_switcher_builder
from JAxtar.utils.chunked_eval import chunked_masked_eval
from qfunction.q_base import QFunction


def _bi_qstar_loop_builder(
    puzzle: Puzzle,
    q_fn: QFunction,
    batch_size: int = 1024,
    max_nodes: int = int(1e6),
    pop_ratio: float = jnp.inf,
    cost_weight: float = 1.0 - 1e-6,
    look_ahead_pruning: bool = True,
    pessimistic_update: bool = True,
    use_backward_q: bool = True,
    backward_mode: str = "auto",
    terminate_on_first_solution: bool = True,
):
    """
    Build the loop components for bidirectional Q* search.

    Args:
        puzzle: Puzzle instance
        q_fn: QFunction instance (used for both directions)
        batch_size: Batch size for parallel processing
        max_nodes: Maximum number of nodes to explore per direction
        pop_ratio: Ratio controlling beam width
        cost_weight: Weight for path cost in f = cost_weight * g + Q(s,a)
        look_ahead_pruning: Enable look-ahead pruning optimization
        pessimistic_update: Use max Q-value for duplicates (True) or min (False)

    Returns:
        Tuple of (init_loop_state, loop_condition, loop_body) functions
    """
    statecls = puzzle.State
    action_size = puzzle.action_size
    dist_sign = -1.0 if pessimistic_update else 1.0
    denom = max(1, puzzle.action_size // 2)
    min_pop = max(1, MIN_BATCH_SIZE // denom)

    variable_q_batch_switcher = variable_batch_switcher_builder(
        q_fn.batched_q_value,
        max_batch_size=batch_size,
        min_batch_size=MIN_BATCH_SIZE,
        pad_value=jnp.inf,
    )
    has_inverse_action_map = hasattr(puzzle, "inverse_action_map")
    backward_use_edge_q = use_backward_q and (
        backward_mode == "edge_q" or (backward_mode == "auto" and has_inverse_action_map)
    )
    backward_use_value_heuristic = use_backward_q and (
        backward_mode == "value_v" or (backward_mode == "auto" and (not has_inverse_action_map))
    )
    backward_use_q = backward_use_edge_q or backward_use_value_heuristic

    def init_loop_state(
        solve_config: Puzzle.SolveConfig,
        start: Puzzle.State,
        **kwargs: Any,
    ) -> BiLoopStateWithStates:
        """Initialize bidirectional Q* search from start and goal states."""
        bi_result = build_bi_search_result(
            statecls,
            batch_size,
            max_nodes,
            action_size,
            pop_ratio=pop_ratio,
            min_pop=min_pop,
            parant_with_costs=True,
        )

        q_params_forward = q_fn.prepare_q_parameters(solve_config, **kwargs)
        inverse_solveconfig = puzzle.hindsight_transform(solve_config, start)

        if use_backward_q:
            q_params_backward = q_fn.prepare_q_parameters(inverse_solveconfig, **kwargs)
        else:
            q_params_backward = q_params_forward

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

    def _build_eval_deferred_q(
        *,
        is_forward: bool,
        use_q: bool,
        use_value_heuristic: bool,
        can_optimize_bwd: bool,
    ):
        get_neighbours_fn = (
            puzzle.batched_get_neighbours if is_forward else puzzle.batched_get_inverse_neighbours
        )

        def _eval_deferred_q(
            puzzle: Puzzle,
            search_result: Any,
            solve_config: Puzzle.SolveConfig,
            q_params: Any,
            states: Puzzle.State,
            costs: chex.Array,
            filled_tiles: chex.Array,
            filled: chex.Array,
            look_ahead_pruning: bool,
            cost_weight: float,
        ) -> tuple[chex.Array, chex.Array, chex.Array]:
            sr_batch_size = search_result.batch_size
            flattened_filled_tiles = filled_tiles.flatten()

            if look_ahead_pruning:
                neighbour_look_ahead, ncosts = get_neighbours_fn(solve_config, states, filled)
                look_ahead_costs = (costs + ncosts).astype(KEY_DTYPE)
                flattened_neighbour_look_ahead = neighbour_look_ahead.flatten()
                flattened_look_ahead_costs = look_ahead_costs.flatten().astype(KEY_DTYPE)

                if use_value_heuristic:
                    current_hash_idxs, found = search_result.hashtable.lookup_parallel(
                        flattened_neighbour_look_ahead, flattened_filled_tiles
                    )
                    old_costs = search_result.get_cost(current_hash_idxs)

                    candidate_mask = flattened_filled_tiles & jnp.logical_or(
                        ~found, jnp.less(flattened_look_ahead_costs, old_costs)
                    )
                    unique_mask = xnp.unique_mask(
                        flattened_neighbour_look_ahead,
                        flattened_look_ahead_costs,
                        candidate_mask,
                    )
                    optimal_mask_flat = unique_mask & candidate_mask
                    optimal_mask = optimal_mask_flat.reshape(action_size, sr_batch_size)

                    found_reshaped = found.reshape(action_size, sr_batch_size)
                    old_dists = search_result.get_dist(current_hash_idxs).reshape(
                        action_size, sr_batch_size
                    )
                    need_compute = optimal_mask & ~found_reshaped

                    flat_states = neighbour_look_ahead.flatten()
                    flat_need_compute = need_compute.flatten()

                    computed_heuristic_vals = chunked_masked_eval(
                        lambda s, m: jnp.min(variable_q_batch_switcher(q_params, s, m), axis=-1),
                        flat_states,
                        flat_need_compute,
                        action_size,
                        sr_batch_size,
                    ).reshape(action_size, sr_batch_size)

                    dists = jnp.where(found_reshaped, old_dists, computed_heuristic_vals)
                    dists = jnp.where(filled_tiles, dists, jnp.inf).astype(KEY_DTYPE)
                    neighbour_keys = (cost_weight * look_ahead_costs + dists).astype(KEY_DTYPE)
                else:
                    if use_q:
                        q_vals = variable_q_batch_switcher(q_params, states, filled)
                        q_vals = q_vals.transpose().astype(KEY_DTYPE)
                        if (not is_forward) and can_optimize_bwd:
                            q_vals = q_vals[puzzle.inverse_action_map, :]
                    else:
                        q_vals = ncosts.astype(KEY_DTYPE)

                    dists_flat = q_vals.flatten()
                    if use_q:
                        distinct_score = (
                            flattened_look_ahead_costs + dist_sign * 1e-5 * dists_flat
                        ).astype(KEY_DTYPE)
                    else:
                        distinct_score = flattened_look_ahead_costs

                    unique_mask = xnp.unique_mask(
                        flattened_neighbour_look_ahead,
                        distinct_score,
                        flattened_filled_tiles,
                    )
                    current_hash_idxs, found = search_result.hashtable.lookup_parallel(
                        flattened_neighbour_look_ahead, unique_mask
                    )
                    old_costs = search_result.get_cost(current_hash_idxs)
                    old_dists = search_result.get_dist(current_hash_idxs)

                    if use_q:
                        step_cost = ncosts.flatten().astype(KEY_DTYPE)
                        q_old = old_dists.astype(KEY_DTYPE) + step_cost
                        if pessimistic_update:
                            q_old_for_max = jnp.where(found, q_old, -jnp.inf)
                            dists_flat = jnp.maximum(dists_flat, q_old_for_max)
                        else:
                            q_old_for_min = jnp.where(found, q_old, jnp.inf)
                            dists_flat = jnp.minimum(dists_flat, q_old_for_min)

                    better_cost_mask = jnp.less(flattened_look_ahead_costs, old_costs)
                    optimal_mask_flat = unique_mask & jnp.logical_or(~found, better_cost_mask)
                    optimal_mask = optimal_mask_flat.reshape(action_size, sr_batch_size)

                    dists = dists_flat.reshape(action_size, sr_batch_size).astype(KEY_DTYPE)
                    neighbour_keys = (cost_weight * costs + q_vals).astype(KEY_DTYPE)
            else:
                if use_value_heuristic:
                    q_vals = variable_q_batch_switcher(q_params, states, filled)
                    v_vals = jnp.min(q_vals, axis=-1).astype(KEY_DTYPE)
                    v_vals = jnp.where(filled, v_vals, jnp.inf)
                    dists = jnp.tile(v_vals[jnp.newaxis, :], (action_size, 1)).astype(KEY_DTYPE)
                else:
                    if use_q:
                        q_vals = variable_q_batch_switcher(q_params, states, filled)
                        q_vals = q_vals.transpose().astype(KEY_DTYPE)
                        if (not is_forward) and can_optimize_bwd:
                            q_vals = q_vals[puzzle.inverse_action_map, :]
                    else:
                        q_vals = jnp.zeros_like(costs, dtype=KEY_DTYPE)
                    dists = q_vals.astype(KEY_DTYPE)

                optimal_mask = flattened_filled_tiles.reshape(action_size, sr_batch_size)
                neighbour_keys = (cost_weight * costs + dists).astype(KEY_DTYPE)

            neighbour_keys = jnp.where(filled_tiles, neighbour_keys, jnp.inf)
            return neighbour_keys, dists, optimal_mask

        return _eval_deferred_q

    eval_forward = _build_eval_deferred_q(
        is_forward=True,
        use_q=True,
        use_value_heuristic=False,
        can_optimize_bwd=False,
    )
    eval_backward = _build_eval_deferred_q(
        is_forward=False,
        use_q=backward_use_q,
        use_value_heuristic=backward_use_value_heuristic,
        can_optimize_bwd=backward_use_edge_q,
    )
    expand_forward_direction = build_bi_deferred_expand_direction(
        puzzle,
        cost_weight,
        look_ahead_pruning,
        eval_forward,
        is_forward=True,
        use_heuristic_in_pop=False,
    )
    expand_backward_direction = build_bi_deferred_expand_direction(
        puzzle,
        cost_weight,
        look_ahead_pruning,
        eval_backward,
        is_forward=False,
        use_heuristic_in_pop=backward_use_value_heuristic,
    )

    def loop_body(loop_state: BiLoopStateWithStates) -> BiLoopStateWithStates:
        """Main loop body for bidirectional Q*."""
        bi_result = loop_state.bi_result
        solve_config = loop_state.solve_config
        inverse_solveconfig = loop_state.inverse_solveconfig

        fwd_not_full = bi_result.forward.generated_size < bi_result.forward.capacity
        bwd_not_full = bi_result.backward.generated_size < bi_result.backward.capacity

        def _expand_forward(bi_result):
            return expand_forward_direction(
                bi_result,
                solve_config,
                inverse_solveconfig,
                loop_state.params_forward,
                loop_state.current_forward,
                loop_state.states_forward,
                loop_state.filled_forward,
            )

        def _expand_backward(bi_result):
            return expand_backward_direction(
                bi_result,
                solve_config,
                inverse_solveconfig,
                loop_state.params_backward,
                loop_state.current_backward,
                loop_state.states_backward,
                loop_state.filled_backward,
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
    spec: SearchBuildSpec = DEFAULT_SEARCH_BUILD_SPEC,
    *,
    look_ahead_pruning: bool = True,
    pessimistic_update: bool = True,
    backward_mode: str = "auto",
    terminate_on_first_solution: bool = True,
    unsafe_allow_nonadmissible: bool = False,
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
        unsafe_allow_nonadmissible: If True, allow `terminate_on_first_solution=False`
            when backward mode is not "dijkstra". This can violate optimality claims.

    Returns:
        A JIT-compiled function that performs bidirectional Q* search
    """
    import warnings

    _require_no_workload_signature(spec)
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
        if not unsafe_allow_nonadmissible:
            raise ValueError(
                "bi_qstar with terminate_on_first_solution=False requires an admissible lower "
                "bound consistent with PQ keys. This is not guaranteed for learned/approximate Q "
                "unless backward_mode='dijkstra'. Set unsafe_allow_nonadmissible=True only if you "
                "accept potentially non-optimal early termination guarantees."
            )
        warnings.warn(
            "bi_qstar running with terminate_on_first_solution=False and non-dijkstra "
            "backward mode under unsafe_allow_nonadmissible=True. Optimality is not guaranteed.",
            UserWarning,
        )

    use_backward_q = (not q_fn.is_fixed) and backward_mode != "dijkstra"
    init_loop_state, loop_condition, loop_body = _bi_qstar_loop_builder(
        puzzle,
        q_fn,
        batch_size,
        max_nodes,
        spec.pop_ratio,
        spec.cost_weight,
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
        loop_state = init_loop_state(solve_config, start, **kwargs)
        loop_state = jax.lax.while_loop(loop_condition, loop_body, loop_state)

        inverse_solveconfig = loop_state.inverse_solveconfig
        bi_result = loop_state.bi_result

        # Materialize meeting hashidxs if the best meeting was found via edge-only tracking.
        bi_result = materialize_meeting_point_hashidxs(
            bi_result,
            puzzle,
            solve_config,
            inverse_solveconfig=inverse_solveconfig,
        )

        return stamp_bi_solved_from_meeting(bi_result)

    return compile_search_builder(bi_qstar, puzzle, spec.show_compile_time, spec.warmup_inputs)
