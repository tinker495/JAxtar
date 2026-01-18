"""
Iterative Deepening Q* (ID-Q*) Search Implementation
"""

import jax
import jax.numpy as jnp
import xtructure.numpy as xnp
from puxle import Puzzle

from JAxtar.annotate import KEY_DTYPE, MIN_BATCH_SIZE
from JAxtar.id_stars.id_frontier import (
    ACTION_PAD,
    IDFrontier,
    build_flat_children,
    build_id_node_batch,
)
from JAxtar.id_stars.search_base import (
    IDLoopState,
    IDSearchBase,
    apply_non_backtracking,
    build_outer_loop,
    finalize_builder,
)
from JAxtar.utils.batch_switcher import variable_batch_switcher_builder
from qfunction.q_base import QFunction


def _id_qstar_frontier_builder(
    puzzle: Puzzle,
    q_fn: QFunction,
    batch_size: int = 1024,
    cost_weight: float = 1.0,
    non_backtracking_steps: int = 3,
    max_path_len: int = 256,
):
    """
    Q-optimized frontier builder for ID-Q*.
    Evaluates Q on parent states and uses the action scores to rank children.
    """
    action_size = puzzle.action_size
    flat_size = action_size * batch_size
    statecls = puzzle.State
    empty_trail_flat = statecls.default((flat_size, 0))

    if non_backtracking_steps < 0:
        raise ValueError("non_backtracking_steps must be non-negative")
    non_backtracking_steps = int(non_backtracking_steps)
    trail_indices = jnp.arange(non_backtracking_steps, dtype=jnp.int32)

    IDNodeBatch = build_id_node_batch(statecls, non_backtracking_steps, max_path_len)

    variable_q_parent_switcher = variable_batch_switcher_builder(
        q_fn.batched_q_value,
        max_batch_size=batch_size,
        min_batch_size=MIN_BATCH_SIZE,
        pad_value=jnp.inf,
    )

    def generate_frontier(
        solve_config: Puzzle.SolveConfig, start: Puzzle.State, **kwargs
    ) -> IDFrontier:
        if "h_params" in kwargs:
            q_params = kwargs["h_params"]
        else:
            q_params = q_fn.prepare_q_parameters(solve_config, **kwargs)

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
                not_solved,
                jnp.logical_and(
                    within_limit,
                    jnp.logical_and(has_capacity, has_nodes),
                ),
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

            flat_parent_indices = jnp.tile(jnp.arange(batch_size, dtype=jnp.int32), action_size)
            flat_parent_indices = jnp.where(flat_valid, flat_parent_indices, -1)
            flat_root_indices = flat_parent_indices

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

            q_vals = variable_q_parent_switcher(q_params, states, valid).astype(KEY_DTYPE)
            q_vals = jnp.where(valid[:, None], q_vals, jnp.inf)
            q_vals = jnp.maximum(0.0, q_vals)  # Ensure non-negative Q-values
            q_vals = q_vals.transpose()
            f_vals = (cost_weight * gs[jnp.newaxis, :] + q_vals).astype(KEY_DTYPE)
            flat_f = f_vals.reshape((flat_size,))

            unique_mask = xnp.unique_mask(
                flat_states,
                key=flat_g,
                filled=flat_valid,
            )
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

            f_safe = jnp.where(flat_valid, jnp.nan_to_num(flat_f, nan=1e5, posinf=1e5), jnp.inf)

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


def _id_qstar_loop_builder(
    puzzle: Puzzle,
    q_fn: QFunction,
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
    if non_backtracking_steps < 0:
        raise ValueError("non_backtracking_steps must be non-negative")
    non_backtracking_steps = int(non_backtracking_steps)
    trail_indices = jnp.arange(non_backtracking_steps, dtype=jnp.int32)
    action_ids = jnp.arange(action_size, dtype=jnp.int32)
    frontier_actions = jnp.full((batch_size,), ACTION_PAD, dtype=jnp.int32)

    IDNodeBatch = build_id_node_batch(statecls, non_backtracking_steps, max_path_len)

    variable_q_parent_switcher = variable_batch_switcher_builder(
        q_fn.batched_q_value,
        max_batch_size=batch_size,
        min_batch_size=MIN_BATCH_SIZE,
        pad_value=jnp.inf,
    )

    def _min_q(q_params, states, valid_mask):
        q_vals = variable_q_parent_switcher(q_params, states, valid_mask).astype(KEY_DTYPE)
        q_vals = jnp.where(valid_mask[:, None], q_vals, jnp.inf)
        q_vals = jnp.maximum(0.0, q_vals)  # Ensure non-negative Q-values
        return jnp.min(q_vals, axis=-1)

    generate_frontier = _id_qstar_frontier_builder(
        puzzle,
        q_fn,
        batch_size=batch_size,
        cost_weight=cost_weight,
        non_backtracking_steps=non_backtracking_steps,
        max_path_len=max_path_len,
    )

    def init_loop_state(solve_config: Puzzle.SolveConfig, start: Puzzle.State, **kwargs):
        search_result = IDSearchBase.build(
            statecls,
            capacity=max_nodes,
            action_size=action_size,
            non_backtracking_steps=non_backtracking_steps,
            max_path_len=max_path_len,
        )

        q_parameters = q_fn.prepare_q_parameters(solve_config, **kwargs)

        frontier = generate_frontier(solve_config, start, h_params=q_parameters)

        search_result, frontier = search_result.initialize_from_frontier(
            frontier,
            cost_weight,
            _min_q,
            q_parameters,
            frontier_actions,
        )

        return IDLoopState(
            search_result=search_result,
            solve_config=solve_config,
            params=q_parameters,
            frontier=frontier,
        )

    def inner_cond(loop_state: IDLoopState):
        sr = loop_state.search_result
        return jnp.logical_and(sr.stack_ptr > 0, ~sr.solved)

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

        q_vals = variable_q_parent_switcher(params, parents, valid_mask).astype(KEY_DTYPE)
        q_vals = jnp.where(valid_mask[:, None], q_vals, jnp.inf)
        q_vals = jnp.maximum(0.0, q_vals)
        q_vals = q_vals.transpose()
        f_vals = (cost_weight * parent_costs[jnp.newaxis, :] + q_vals).astype(KEY_DTYPE)
        flat_f = f_vals.reshape((flat_size,))

        # --- Optimization: Pruning by Bound (f > bound) ---
        active_bound = sr.bound
        f_prune_mask = flat_f > active_bound + 1e-6

        valid_f_pruned = jnp.logical_and(flat_valid, f_prune_mask)
        min_f_pruned = jnp.min(jnp.where(valid_f_pruned, flat_f, jnp.inf)).astype(KEY_DTYPE)

        # Update next_bound in SR
        new_next_bound_f = jnp.minimum(sr.next_bound, min_f_pruned).astype(KEY_DTYPE)
        sr = sr.replace(next_bound=new_next_bound_f)
        sr_solved = sr_solved.replace(next_bound=new_next_bound_f)

        flat_valid = jnp.logical_and(flat_valid, flat_f <= active_bound + 1e-6)

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

        flat_valid = apply_non_backtracking(
            flat_neighbours,
            parents,
            parent_trails,
            parent_depths,
            flat_valid,
            non_backtracking_steps,
            action_size,
            flat_size,
            trail_indices,
            batch_size,
        )

        flat_action_history = jnp.where(
            flat_valid[:, None], flat_action_history, jnp.full_like(flat_action_history, ACTION_PAD)
        )

        return_sr = jax.lax.cond(
            any_solved,
            lambda s: s,
            lambda s: _expand_step(
                s,
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
            ),
            sr_solved,
        )

        return loop_state.replace(search_result=return_sr)

    def _expand_step(
        sr,
        states,
        gs,
        depths,
        actions,
        trails,
        action_histories,
        valid,
        fs,
        parent_indices,
        root_indices,
    ):
        flat_batch = IDNodeBatch(
            state=states,
            cost=gs,
            depth=depths,
            action=actions,
            trail=trails,
            action_history=action_histories,
            parent_index=parent_indices,
            root_index=root_indices,
        )

        return sr.expand_and_push(flat_batch, fs, valid, update_next_bound=False)

    outer_cond, outer_body = build_outer_loop(inner_cond, inner_body, statecls, frontier_actions)

    return init_loop_state, outer_cond, outer_body


def id_qstar_builder(
    puzzle: Puzzle,
    q_fn: QFunction,
    batch_size: int = 1024,
    max_nodes: int = int(1e6),
    cost_weight: float = 1.0,
    pop_ratio: float = 1.0,
    non_backtracking_steps: int = 0,
    show_compile_time: bool = False,
    max_path_len: int = 256,
):
    init_loop, cond, body = _id_qstar_loop_builder(
        puzzle,
        q_fn,
        batch_size,
        max_nodes,
        cost_weight,
        non_backtracking_steps=non_backtracking_steps,
        max_path_len=max_path_len,
    )

    return finalize_builder(
        puzzle, init_loop, cond, body, name="ID-Q*", show_compile_time=show_compile_time
    )
