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
import xtructure.numpy as xnp
from puxle import Puzzle

from helpers.jax_compile import compile_search_builder
from heuristic.heuristic_base import Heuristic
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
from JAxtar.utils.array_ops import stable_partition_three
from JAxtar.utils.batch_switcher import variable_batch_switcher_builder


def _bi_astar_d_loop_builder(
    puzzle: Puzzle,
    heuristic: Heuristic,
    batch_size: int = 1024,
    max_nodes: int = int(1e6),
    pop_ratio: float = jnp.inf,
    cost_weight: float = 1.0 - 1e-6,
    look_ahead_pruning: bool = True,
    use_backward_heuristic: bool = True,
    backward_value_lookahead: bool = False,
    backward_value_lookahead_k: int | None = None,
    terminate_on_first_solution: bool = True,
):
    """
    Build the loop components for bidirectional A* deferred search.

    Args:
        puzzle: Puzzle instance
        heuristic: Heuristic instance (used for both directions)
        batch_size: Batch size for parallel processing
        max_nodes: Maximum number of nodes to explore per direction
        pop_ratio: Ratio controlling beam width
        cost_weight: Weight for path cost in f = cost_weight * g + h
        look_ahead_pruning: Enable look-ahead pruning optimization

    Returns:
        Tuple of (init_loop_state, loop_condition, loop_body) functions
    """
    statecls = puzzle.State
    action_size = puzzle.action_size
    denom = max(1, puzzle.action_size // 2)
    min_pop = max(1, MIN_BATCH_SIZE // denom)

    variable_heuristic_batch_switcher = variable_batch_switcher_builder(
        heuristic.batched_distance,
        max_batch_size=batch_size,
        min_batch_size=MIN_BATCH_SIZE,
        pad_value=jnp.inf,
    )

    def init_loop_state(
        solve_config: Puzzle.SolveConfig,
        start: Puzzle.State,
        **kwargs: Any,
    ) -> BiLoopStateWithStates:
        """Initialize bidirectional deferred search from start and goal states."""
        bi_result = build_bi_search_result(
            statecls,
            batch_size,
            max_nodes,
            action_size,
            pop_ratio=pop_ratio,
            min_pop=min_pop,
            parant_with_costs=True,
        )

        heuristic_params_forward = heuristic.prepare_heuristic_parameters(solve_config, **kwargs)
        inverse_solveconfig = puzzle.hindsight_transform(solve_config, start)

        if use_backward_heuristic:
            heuristic_params_backward = heuristic.prepare_heuristic_parameters(
                inverse_solveconfig, **kwargs
            )
        else:
            heuristic_params_backward = heuristic_params_forward

        if backward_value_lookahead:
            heuristic_params_backward = (heuristic_params_backward, solve_config)

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

    def _build_eval_deferred_heuristic(
        get_neighbours_fn,
        *,
        use_heuristic: bool,
        enable_backward_value_lookahead: bool,
    ):
        def _eval_deferred_heuristic(
            puzzle: Puzzle,
            search_result: Any,
            solve_config: Puzzle.SolveConfig,
            params: Any,
            states: Puzzle.State,
            costs: chex.Array,
            filled_tiles: chex.Array,
            filled: chex.Array,
            look_ahead_pruning: bool,
            cost_weight: float,
        ) -> tuple[chex.Array, chex.Array, chex.Array]:
            if enable_backward_value_lookahead:
                # params is packed as (heuristic_params, forward_solve_config)
                heuristic_params, forward_solve_config = params
            else:
                heuristic_params = params
                forward_solve_config = None

            action_size = search_result.action_size
            sr_batch_size = search_result.batch_size
            flat_size = action_size * sr_batch_size
            flattened_filled_tiles = filled_tiles.flatten()

            if look_ahead_pruning:
                neighbour_look_a_head, ncosts = get_neighbours_fn(solve_config, states, filled)
                look_a_head_costs = (costs + ncosts).astype(KEY_DTYPE)

                flattened_neighbour_look_head = neighbour_look_a_head.flatten()
                flattened_look_a_head_costs = look_a_head_costs.flatten().astype(KEY_DTYPE)

                current_hash_idxs, found = search_result.hashtable.lookup_parallel(
                    flattened_neighbour_look_head, flattened_filled_tiles
                )
                old_costs = search_result.get_cost(current_hash_idxs)

                candidate_mask = flattened_filled_tiles & jnp.logical_or(
                    ~found, jnp.less(flattened_look_a_head_costs, old_costs)
                )
                unique_mask = xnp.unique_mask(
                    flattened_neighbour_look_head,
                    flattened_look_a_head_costs,
                    candidate_mask,
                )
                optimal_mask_flat = unique_mask & candidate_mask
                optimal_mask = optimal_mask_flat.reshape(action_size, sr_batch_size)

                if use_heuristic:
                    found_reshaped = found.reshape(action_size, sr_batch_size)
                    old_dists = search_result.get_dist(current_hash_idxs).reshape(
                        action_size, sr_batch_size
                    )

                    need_compute = optimal_mask & ~found_reshaped
                    flat_states = neighbour_look_a_head.flatten()
                    flat_need_compute = need_compute.flatten()

                    sorted_indices = stable_partition_three(
                        flat_need_compute,
                        jnp.zeros_like(flat_need_compute, dtype=jnp.bool_),
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
                        jnp.empty((flat_size,), dtype=h_val_sorted.dtype)
                        .at[sorted_indices]
                        .set(h_val_sorted)
                    )
                    computed_heuristic_vals = flat_h_val.reshape(action_size, sr_batch_size)

                    heuristic_vals = jnp.where(
                        found_reshaped,
                        old_dists,
                        computed_heuristic_vals,
                    )
                    heuristic_vals = jnp.where(filled_tiles, heuristic_vals, jnp.inf).astype(
                        KEY_DTYPE
                    )
                else:
                    heuristic_vals = jnp.zeros_like(look_a_head_costs, dtype=KEY_DTYPE)
                    heuristic_vals = jnp.where(filled_tiles, heuristic_vals, jnp.inf)

                if enable_backward_value_lookahead and use_heuristic:
                    base_keys = (cost_weight * look_a_head_costs + heuristic_vals).astype(KEY_DTYPE)
                    flat_keys = base_keys.flatten()
                    masked_flat_keys = jnp.where(optimal_mask_flat, flat_keys, jnp.inf)

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
                        forward_solve_config, sel_states, sel_valid
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

                neighbour_keys = (cost_weight * look_a_head_costs + heuristic_vals).astype(
                    KEY_DTYPE
                )
            else:
                if use_heuristic:
                    heuristic_vals = variable_heuristic_batch_switcher(
                        heuristic_params, states, filled
                    )
                    heuristic_vals = jnp.where(filled, heuristic_vals, jnp.inf)
                    heuristic_vals = jnp.tile(
                        heuristic_vals[jnp.newaxis, :], (action_size, 1)
                    ).astype(KEY_DTYPE)
                else:
                    heuristic_vals = jnp.zeros_like(costs, dtype=KEY_DTYPE)
                    heuristic_vals = jnp.where(filled_tiles, heuristic_vals, jnp.inf)

                neighbour_keys = (cost_weight * costs + heuristic_vals).astype(KEY_DTYPE)
                optimal_mask = flattened_filled_tiles.reshape(action_size, sr_batch_size)

            neighbour_keys = jnp.where(filled_tiles, neighbour_keys, jnp.inf)
            return neighbour_keys, heuristic_vals, optimal_mask

        return _eval_deferred_heuristic

    eval_forward = _build_eval_deferred_heuristic(
        puzzle.batched_get_neighbours,
        use_heuristic=True,
        enable_backward_value_lookahead=False,
    )
    eval_backward = _build_eval_deferred_heuristic(
        puzzle.batched_get_inverse_neighbours,
        use_heuristic=use_backward_heuristic,
        enable_backward_value_lookahead=backward_value_lookahead,
    )

    expand_forward_direction = build_bi_deferred_expand_direction(
        puzzle,
        cost_weight,
        look_ahead_pruning,
        eval_forward,
        is_forward=True,
        use_heuristic_in_pop=True,
    )
    expand_backward_direction = build_bi_deferred_expand_direction(
        puzzle,
        cost_weight,
        look_ahead_pruning,
        eval_backward,
        is_forward=False,
        use_heuristic_in_pop=True,
    )

    def loop_body(loop_state: BiLoopStateWithStates) -> BiLoopStateWithStates:
        """Main loop body for bidirectional A* deferred."""
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


def bi_astar_d_builder(
    puzzle: Puzzle,
    heuristic: Heuristic,
    batch_size: int = 1024,
    max_nodes: int = int(1e6),
    spec: SearchBuildSpec = DEFAULT_SEARCH_BUILD_SPEC,
    *,
    look_ahead_pruning: bool = True,
    experimental_backward_value_lookahead: bool = False,
    experimental_backward_value_lookahead_k: int | None = None,
    terminate_on_first_solution: bool = True,
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
        experimental_backward_value_lookahead: Experimental value-lookahead for
            backward deferred expansion. Disabled by default for stars parity.
        experimental_backward_value_lookahead_k: Number of backward candidates
            to refine when experimental_backward_value_lookahead=True.

    Returns:
        A JIT-compiled function that performs bidirectional A* deferred search
    """
    # Bidirectional deferred requires look_ahead_pruning for correct f-value computation
    # Without it, the termination condition based on get_min_f_value() can be incorrect
    import warnings

    if not look_ahead_pruning:
        warnings.warn(
            "Bidirectional A* deferred requires look_ahead_pruning=True for correct "
            "termination. Forcing look_ahead_pruning=True.",
            UserWarning,
        )
        look_ahead_pruning = True

    if experimental_backward_value_lookahead:
        warnings.warn(
            "experimental_backward_value_lookahead=True enables non-parity "
            "experimental behavior in bi_astar_d backward expansion.",
            UserWarning,
        )

    _require_no_workload_signature(spec)
    use_backward_heuristic = not heuristic.is_fixed
    init_loop_state, loop_condition, loop_body = _bi_astar_d_loop_builder(
        puzzle,
        heuristic,
        batch_size,
        max_nodes,
        spec.pop_ratio,
        spec.cost_weight,
        look_ahead_pruning,
        use_backward_heuristic=use_backward_heuristic,
        backward_value_lookahead=experimental_backward_value_lookahead,
        backward_value_lookahead_k=experimental_backward_value_lookahead_k,
        terminate_on_first_solution=terminate_on_first_solution,
    )

    def bi_astar_d(
        solve_config: Puzzle.SolveConfig,
        start: Puzzle.State,
        **kwargs: Any,
    ) -> BiDirectionalSearchResult:
        """Perform bidirectional A* deferred search."""
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

    return compile_search_builder(bi_astar_d, puzzle, spec.show_compile_time, spec.warmup_inputs)
