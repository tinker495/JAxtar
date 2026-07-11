import jax
import jax.numpy as jnp
import xtructure.numpy as xnp
from puxle import Puzzle

from heuristic.heuristic_base import Heuristic
from JAxtar.annotate import KEY_DTYPE, MIN_BATCH_SIZE
from JAxtar.search_build_spec import (
    DEFAULT_SEARCH_BUILD_SPEC,
    SearchBuildSpec,
    _require_no_workload_signature,
)
from JAxtar.id_stars.id_frontier import (
    ACTION_PAD,
    IDFrontier,
    build_flat_children,
    build_id_node_batch,
    validate_non_backtracking_steps,
)
from JAxtar.id_stars.search_base import (
    IDLoopState,
    IDSearchResult,
    apply_non_backtracking,
    apply_standard_deduplication,
    build_frontier_cond,
    build_inner_cond,
    build_outer_loop,
    finalize_builder,
    merge_frontier_solution,
)
from JAxtar.utils.batch_switcher import variable_batch_switcher_builder
from JAxtar.utils.chunked_eval import chunked_masked_eval


def _build_chunked_heuristic_eval(
    variable_heuristic_fn,
    action_size: int,
    batch_size: int,
):
    """
    Factory function to create a chunked heuristic evaluation function.

    This avoids code duplication between frontier builder and loop builder.
    The returned function delegates to the shared ``chunked_masked_eval`` primitive,
    closing over the heuristic parameters so the primitive stays value-fn generic.
    """

    def _chunked_heuristic_eval(
        h_params: Puzzle.SolveConfig,
        flat_states: Puzzle.State,
        flat_valid: jnp.ndarray,
    ) -> jnp.ndarray:
        return chunked_masked_eval(
            lambda s, m: variable_heuristic_fn(h_params, s, m),
            flat_states,
            flat_valid,
            action_size,
            batch_size,
        )

    return _chunked_heuristic_eval


def _id_astar_frontier_builder(
    puzzle: Puzzle,
    heuristic: Heuristic,
    batch_size: int = 1024,
    cost_weight: float = 1.0,
    non_backtracking_steps: int = 3,
    max_path_len: int = 256,
):
    """
    Returns a function that generates an initial frontier starting from a single state.
    Uses Batched BFS with in-batch deduplication. (Inlined for ID-A*)
    """
    action_size = puzzle.action_size
    flat_size = action_size * batch_size
    statecls = puzzle.State
    empty_trail_flat = statecls.default((flat_size, 0))

    non_backtracking_steps = validate_non_backtracking_steps(non_backtracking_steps)

    IDNodeBatch = build_id_node_batch(statecls, non_backtracking_steps, max_path_len)

    variable_heuristic = variable_batch_switcher_builder(
        heuristic.batched_distance,
        max_batch_size=batch_size,
        min_batch_size=MIN_BATCH_SIZE,
        pad_value=jnp.inf,
    )
    trail_indices = jnp.arange(non_backtracking_steps, dtype=jnp.int32)

    _chunked_heuristic_eval = _build_chunked_heuristic_eval(
        variable_heuristic, action_size, batch_size
    )

    def generate_frontier(
        solve_config: Puzzle.SolveConfig, start: Puzzle.State, **kwargs
    ) -> IDFrontier:
        if "h_params" in kwargs:
            h_params = kwargs["h_params"]
        else:
            h_params = heuristic.prepare_heuristic_parameters(solve_config, **kwargs)

        init_val = IDFrontier.initialize_from_start(
            puzzle,
            solve_config,
            start,
            batch_size,
            non_backtracking_steps,
            max_path_len,
        )

        action_ids = jnp.arange(action_size, dtype=jnp.int32)
        cond_bounded = build_frontier_cond(batch_size)

        def body_bounded(val: tuple[IDFrontier, jnp.int32]):
            frontier, i = val

            states = frontier.states
            gs = frontier.costs
            depth = frontier.depths
            valid = frontier.valid_mask
            trail = frontier.trail
            action_history = frontier.action_history

            neighbours, step_costs = puzzle.batched_get_neighbours(solve_config, states, valid)

            (
                flat_states,
                flat_g,
                flat_depth,
                flat_trail,
                flat_action_history,
                flat_actions,
                flat_valid,
            ) = build_flat_children(
                neighbours,
                step_costs,
                gs,
                depth,
                states,
                trail,
                action_history,
                action_ids,
                action_size,
                batch_size,
                flat_size,
                non_backtracking_steps,
                max_path_len,
                empty_trail_flat,
                valid,
            )

            (
                any_solved,
                found_sol_state,
                found_sol_cost,
                found_sol_actions,
                _,
            ) = IDSearchResult.detect_solution(
                puzzle,
                solve_config,
                flat_states,
                flat_g,
                flat_action_history,
                flat_valid,
            )

            (new_solved, new_sol_state, new_sol_cost, new_sol_actions,) = merge_frontier_solution(
                frontier,
                any_solved,
                found_sol_state,
                found_sol_cost,
                found_sol_actions,
            )

            unique_mask = xnp.unique_mask(flat_states, key=flat_g, filled=flat_valid)
            flat_valid = jnp.logical_and(flat_valid, unique_mask)

            flat_valid = apply_non_backtracking(
                flat_states,
                states,
                trail,
                depth,
                flat_valid,
                non_backtracking_steps,
                action_size,
                flat_size,
                trail_indices,
                batch_size,
            )

            # chunked_masked_eval already clamps values to >= 0.
            flat_h = _chunked_heuristic_eval(h_params, flat_states, flat_valid)
            flat_f = (cost_weight * flat_g + flat_h).astype(KEY_DTYPE)
            f_safe = jnp.where(flat_valid, jnp.nan_to_num(flat_f, nan=1e5, posinf=1e5), jnp.inf)

            flat_parent_indices = jnp.tile(jnp.arange(batch_size, dtype=jnp.int32), action_size)
            flat_parent_indices = jnp.where(flat_valid, flat_parent_indices, -1)
            flat_root_indices = flat_parent_indices

            flat_batch = IDNodeBatch(
                state=flat_states,
                cost=flat_g,
                depth=flat_depth,
                trail=flat_trail,
                action_history=flat_action_history,
                action=flat_actions,
                parent_index=flat_parent_indices,
                root_index=flat_root_indices,
            )

            new_frontier = frontier.select_top_k(
                flat_batch,
                flat_valid,
                f_safe,
                batch_size,
                new_solved,
                new_sol_state,
                new_sol_cost,
                new_sol_actions,
            )
            return (new_frontier, i + 1)

        init_loop = (init_val, jnp.array(0, dtype=jnp.int32))
        final_val = jax.lax.while_loop(cond_bounded, body_bounded, init_loop)
        final_frontier, _ = final_val

        return final_frontier

    return generate_frontier


def _id_astar_loop_builder(
    puzzle: Puzzle,
    heuristic: Heuristic,
    batch_size: int = 1024,
    max_nodes: int = int(1e6),
    cost_weight: float = 1.0,
    non_backtracking_steps: int = 0,
    max_path_len: int = 256,
):
    statecls = puzzle.State
    action_size = puzzle.action_size
    flat_size = action_size * batch_size
    empty_trail_flat = statecls.default((flat_size, 0))
    non_backtracking_steps = validate_non_backtracking_steps(non_backtracking_steps)
    trail_indices = jnp.arange(non_backtracking_steps, dtype=jnp.int32)
    action_ids = jnp.arange(action_size, dtype=jnp.int32)
    frontier_actions = jnp.full((batch_size,), ACTION_PAD, dtype=jnp.int32)

    IDNodeBatch = build_id_node_batch(statecls, non_backtracking_steps, max_path_len)

    variable_heuristic_batch_switcher = variable_batch_switcher_builder(
        heuristic.batched_distance,
        max_batch_size=batch_size,
        min_batch_size=MIN_BATCH_SIZE,
        pad_value=jnp.inf,
    )

    generate_frontier = _id_astar_frontier_builder(
        puzzle,
        heuristic,
        batch_size,
        cost_weight=cost_weight,
        non_backtracking_steps=non_backtracking_steps,
        max_path_len=max_path_len,
    )

    _chunked_heuristic_eval = _build_chunked_heuristic_eval(
        variable_heuristic_batch_switcher,
        action_size,
        batch_size,
    )

    def init_loop_state(solve_config: Puzzle.SolveConfig, start: Puzzle.State, **kwargs):
        search_result: IDSearchResult = IDSearchResult.build(
            statecls,
            max_nodes,
            action_size,
            non_backtracking_steps,
            max_path_len,
        )
        heuristic_parameters = heuristic.prepare_heuristic_parameters(solve_config, **kwargs)

        frontier = generate_frontier(solve_config, start, h_params=heuristic_parameters, **kwargs)

        search_result, frontier = search_result.initialize_from_frontier(
            frontier,
            cost_weight,
            variable_heuristic_batch_switcher,
            heuristic_parameters,
            frontier_actions,
        )

        return IDLoopState(
            search_result=search_result,
            solve_config=solve_config,
            params=heuristic_parameters,
            frontier=frontier,
        )

    # -----------------------------------------------------------------------
    # INNER LOOP: Standard Batched DFS with Pruning by Bound
    # -----------------------------------------------------------------------
    inner_cond = build_inner_cond()

    def inner_body(loop_state: IDLoopState):
        sr = loop_state.search_result
        solve_config = loop_state.solve_config
        params = loop_state.params

        (
            sr,
            sr_solved,
            any_solved,
            parents,
            parent_costs,
            parent_depths,
            parent_trails,
            parent_action_histories,
            valid_mask,
            parent_trace_indices,
            parent_root_indices,
            neighbours,
            step_costs,
        ) = sr.prepare_for_expansion(puzzle, solve_config, batch_size)

        (
            flat_neighbours,
            flat_g,
            flat_depth,
            flat_trail,
            flat_action_history,
            flat_actions,
            flat_valid,
        ) = build_flat_children(
            neighbours,
            step_costs,
            parent_costs,
            parent_depths,
            parents,
            parent_trails,
            parent_action_histories,
            action_ids,
            action_size,
            batch_size,
            flat_size,
            non_backtracking_steps,
            max_path_len,
            empty_trail_flat,
            valid_mask,
        )

        flat_valid = jnp.logical_and(flat_valid, flat_depth <= max_path_len)

        flat_parent_indices = jnp.tile(parent_trace_indices, action_size)
        flat_parent_indices = jnp.where(flat_valid, flat_parent_indices, -1)
        flat_root_indices = jnp.tile(parent_root_indices, action_size)
        flat_root_indices = jnp.where(flat_valid, flat_root_indices, -1)

        # --- Optimization: Deduplication ---
        flat_valid = apply_standard_deduplication(
            flat_neighbours,
            flat_g,
            flat_valid,
            parents,
            parent_trails,
            parent_depths,
            non_backtracking_steps,
            action_size,
            flat_size,
            trail_indices,
            batch_size,
        )

        flat_action_history = jnp.where(
            flat_valid[:, None],
            flat_action_history,
            jnp.full_like(flat_action_history, ACTION_PAD),
        )
        # ----------------------------------------------------

        # Heuristic Evaluation (No Caching - Re-compute Always)
        # chunked_masked_eval already clamps values to >= 0.
        flat_h = _chunked_heuristic_eval(params, flat_neighbours, flat_valid)
        flat_f = (cost_weight * flat_g + flat_h).astype(KEY_DTYPE)

        flat_batch = IDNodeBatch(
            state=flat_neighbours,
            cost=flat_g,
            depth=flat_depth,
            action=flat_actions,
            trail=flat_trail,
            action_history=flat_action_history,
            parent_index=flat_parent_indices,
            root_index=flat_root_indices,
        )

        # Unconditional masked push instead of lax.cond: the conditional forces
        # XLA pass-through copies of the whole packed stack store (299MB per
        # iteration at 1e6 nodes) plus a host predicate sync, while an all-False
        # mask makes expand_and_push a bit-exact no-op (scatter drops every row,
        # next_bound takes min with inf, ptr/count advance by zero).
        push_valid = jnp.logical_and(flat_valid, jnp.logical_not(any_solved))
        return_sr = sr_solved.expand_and_push(
            flat_batch, flat_f, push_valid, update_next_bound=True
        )

        return loop_state.replace(search_result=return_sr)

    outer_cond, outer_body = build_outer_loop(inner_cond, inner_body, statecls, frontier_actions)

    return init_loop_state, outer_cond, outer_body


def id_astar_builder(
    puzzle: Puzzle,
    heuristic: Heuristic,
    batch_size: int = 1024,
    max_nodes: int = int(1e6),
    spec: SearchBuildSpec = DEFAULT_SEARCH_BUILD_SPEC,
    *,
    non_backtracking_steps: int = 0,
    max_path_len: int = 256,
):
    _require_no_workload_signature(spec)
    init_loop, cond, body = _id_astar_loop_builder(
        puzzle,
        heuristic,
        batch_size,
        max_nodes,
        spec.cost_weight,
        non_backtracking_steps=non_backtracking_steps,
        max_path_len=max_path_len,
    )

    return finalize_builder(
        puzzle,
        init_loop,
        cond,
        body,
        name="ID-A*",
        show_compile_time=spec.show_compile_time,
        warmup_inputs=spec.warmup_inputs,
    )
