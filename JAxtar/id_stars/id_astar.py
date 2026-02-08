import jax
import jax.numpy as jnp
import xtructure.numpy as xnp
from puxle import Puzzle

from heuristic.heuristic_base import Heuristic
from JAxtar.annotate import KEY_DTYPE, MIN_BATCH_SIZE
from JAxtar.id_stars.id_frontier import (
    ACTION_PAD,
    IDFrontier,
    build_flat_children,
    build_id_node_batch,
    validate_non_backtracking_steps,
)
from JAxtar.id_stars.search_base import (
    IDLoopState,
    IDSearchBase,
    apply_non_backtracking,
    build_inner_cond,
    build_outer_loop,
    expand_and_push_flat_batch,
    finalize_builder,
)
from JAxtar.utils.array_ops import stable_partition_three
from JAxtar.utils.batch_switcher import variable_batch_switcher_builder


def _build_chunked_heuristic_eval(
    variable_heuristic_fn,
    flat_indices: jnp.ndarray,
    action_size: int,
    batch_size: int,
    flat_size: int,
):
    """
    Factory function to create a chunked heuristic evaluation function.

    This avoids code duplication between frontier builder and loop builder.
    The returned function evaluates heuristics in chunks for memory efficiency.
    """

    def _chunked_heuristic_eval(
        h_params: Puzzle.SolveConfig,
        flat_states: Puzzle.State,
        flat_valid: jnp.ndarray,
    ) -> jnp.ndarray:
        sorted_idx = stable_partition_three(flat_valid, jnp.zeros_like(flat_valid, dtype=jnp.bool_))
        sorted_states = xnp.take(flat_states, sorted_idx, axis=0)
        sorted_mask = flat_valid[sorted_idx]

        chunk_states = xnp.reshape(sorted_states, (action_size, batch_size))
        chunk_mask = sorted_mask.reshape((action_size, batch_size))

        def _compute(_, inputs):
            states_slice, mask_slice = inputs

            def _calc(_):
                vals = variable_heuristic_fn(h_params, states_slice, mask_slice).astype(KEY_DTYPE)
                vals = jnp.maximum(0.0, vals)  # Ensure non-negative heuristic
                return jnp.where(mask_slice, vals, jnp.inf)

            return None, jax.lax.cond(
                jnp.any(mask_slice),
                _calc,
                lambda _: jnp.full((batch_size,), jnp.inf, dtype=KEY_DTYPE),
                None,
            )

        _, chunk_vals = jax.lax.scan(_compute, None, (chunk_states, chunk_mask))
        sorted_vals = chunk_vals.reshape((flat_size,))
        flat_vals = jnp.full((flat_size,), jnp.inf, dtype=KEY_DTYPE)
        flat_vals = flat_vals.at[sorted_idx].set(sorted_vals)
        return flat_vals

    return _chunked_heuristic_eval


def _id_astar_frontier_builder(
    puzzle: Puzzle,
    heuristic: Heuristic,
    batch_size: int = 1024,
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
    flat_indices = jnp.arange(flat_size, dtype=jnp.int32)
    trail_indices = jnp.arange(non_backtracking_steps, dtype=jnp.int32)

    _chunked_heuristic_eval = _build_chunked_heuristic_eval(
        variable_heuristic, flat_indices, action_size, batch_size, flat_size
    )

    def generate_frontier(
        solve_config: Puzzle.SolveConfig, start: Puzzle.State, **kwargs
    ) -> IDFrontier:
        if "h_params" in kwargs:
            h_params = kwargs["h_params"]
        else:
            h_params = heuristic.prepare_heuristic_parameters(solve_config, **kwargs)

        init_val = IDFrontier.initialize_from_start(
            puzzle, solve_config, start, batch_size, non_backtracking_steps, max_path_len
        )

        MAX_FRONTIER_STEPS = 100

        def cond_bounded(val: tuple[IDFrontier, jnp.int32]):
            frontier, i = val
            num_valid = jnp.sum(frontier.valid_mask)
            has_capacity = num_valid < batch_size
            has_nodes = num_valid > 0
            within_limit = i < MAX_FRONTIER_STEPS
            not_solved = ~frontier.solved
            return jnp.logical_and(
                not_solved, jnp.logical_and(within_limit, jnp.logical_and(has_capacity, has_nodes))
            )

        action_ids = jnp.arange(action_size, dtype=jnp.int32)

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
            ) = IDSearchBase.detect_solution(
                puzzle, solve_config, flat_states, flat_g, flat_action_history, flat_valid
            )

            new_solved = jnp.logical_or(frontier.solved, any_solved)
            new_sol_state = jax.lax.cond(
                any_solved, lambda _: found_sol_state, lambda _: frontier.solution_state, None
            )
            new_sol_cost = jax.lax.cond(
                any_solved, lambda _: found_sol_cost, lambda _: frontier.solution_cost, None
            )
            new_sol_actions = jax.lax.cond(
                any_solved,
                lambda _: found_sol_actions,
                lambda _: frontier.solution_actions_arr,
                None,
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

            flat_h = _chunked_heuristic_eval(h_params, flat_states, flat_valid)
            flat_h = jnp.maximum(0.0, flat_h)  # Ensure non-negative
            flat_f = (flat_g + flat_h).astype(KEY_DTYPE)
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
        non_backtracking_steps=non_backtracking_steps,
        max_path_len=max_path_len,
    )

    flat_indices = jnp.arange(flat_size, dtype=jnp.int32)

    _chunked_heuristic_eval = _build_chunked_heuristic_eval(
        variable_heuristic_batch_switcher, flat_indices, action_size, batch_size, flat_size
    )

    def init_loop_state(solve_config: Puzzle.SolveConfig, start: Puzzle.State, **kwargs):
        search_result: IDSearchBase = IDSearchBase.build(
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
        sr, flat_valid = sr.apply_standard_deduplication(
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
            flat_valid[:, None], flat_action_history, jnp.full_like(flat_action_history, ACTION_PAD)
        )
        # ----------------------------------------------------

        # Heuristic Evaluation (No Caching - Re-compute Always)
        flat_h = _chunked_heuristic_eval(params, flat_neighbours, flat_valid)
        flat_h = jnp.maximum(0.0, flat_h)

        flat_f = (cost_weight * flat_g + flat_h).astype(KEY_DTYPE)

        # 4. Expansion
        return_sr = jax.lax.cond(
            any_solved,
            lambda s: s,
            lambda s: expand_and_push_flat_batch(
                s,
                IDNodeBatch,
                flat_neighbours,
                flat_g,
                flat_depth,
                flat_actions,
                flat_trail,
                flat_action_history,
                flat_valid,
                flat_f,
                flat_parent_indices,
                flat_root_indices,
                update_next_bound=True,
            ),
            sr_solved,
        )

        return loop_state.replace(search_result=return_sr)

    outer_cond, outer_body = build_outer_loop(inner_cond, inner_body, statecls, frontier_actions)

    return init_loop_state, outer_cond, outer_body


def id_astar_builder(
    puzzle: Puzzle,
    heuristic: Heuristic,
    batch_size: int = 1024,
    max_nodes: int = int(1e6),
    cost_weight: float = 1.0,
    pop_ratio: float = 1.0,
    non_backtracking_steps: int = 0,
    show_compile_time: bool = False,
    max_path_len: int = 256,
    warmup_inputs: tuple[Puzzle.SolveConfig, Puzzle.State] | None = None,
):
    init_loop, cond, body = _id_astar_loop_builder(
        puzzle,
        heuristic,
        batch_size,
        max_nodes,
        cost_weight,
        non_backtracking_steps=non_backtracking_steps,
        max_path_len=max_path_len,
    )

    return finalize_builder(
        puzzle,
        init_loop,
        cond,
        body,
        name="ID-A*",
        show_compile_time=show_compile_time,
        warmup_inputs=warmup_inputs,
    )
