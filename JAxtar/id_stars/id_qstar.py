"""
Iterative Deepening Q* (ID-Q*) Search Implementation
"""

import time

import jax
import jax.numpy as jnp
import xtructure.numpy as xnp
from puxle import Puzzle

from JAxtar.annotate import KEY_DTYPE, MIN_BATCH_SIZE
from JAxtar.id_stars.search_base import IDFrontier, IDLoopState, IDSearchResult
from JAxtar.utils.batch_switcher import variable_batch_switcher_builder
from qfunction.q_base import QFunction


def _id_qstar_frontier_builder(
    puzzle: Puzzle,
    q_fn: QFunction,
    batch_size: int = 1024,
    cost_weight: float = 1.0,
):
    """
    Q-optimized frontier builder for ID-Q*.
    Evaluates Q on parent states and uses the action scores to rank children.
    """
    action_size = puzzle.action_size

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

        start_reshaped = xnp.expand_dims(start, axis=0)
        root_solved = puzzle.batched_is_solved(solve_config, start_reshaped)[0]

        start_padded = xnp.pad(start_reshaped, ((0, batch_size - 1),), mode="constant")

        costs_padded = jnp.full((batch_size,), jnp.inf, dtype=KEY_DTYPE)
        costs_padded = costs_padded.at[0].set(0.0)
        depths_padded = jnp.full((batch_size,), 0, dtype=jnp.int32)
        valid_padded = jnp.zeros((batch_size,), dtype=jnp.bool_)
        valid_padded = valid_padded.at[0].set(True)

        solution_state = start_reshaped
        solution_cost = jax.lax.cond(
            root_solved,
            lambda _: jnp.array(0, dtype=KEY_DTYPE),
            lambda _: jnp.array(jnp.inf, dtype=KEY_DTYPE),
            None,
        )

        init_val = IDFrontier(
            states=start_padded,
            costs=costs_padded,
            depths=depths_padded,
            valid_mask=valid_padded,
            solved=root_solved,
            solution_state=solution_state,
            solution_cost=solution_cost,
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

        def body_bounded(val: tuple[IDFrontier, jnp.int32]):
            frontier, i = val

            states = frontier.states
            gs = frontier.costs
            depth = frontier.depths
            valid = frontier.valid_mask

            neighbours, step_costs = puzzle.batched_get_neighbours(solve_config, states, valid)

            child_g = (gs[jnp.newaxis, :] + step_costs).astype(KEY_DTYPE)
            child_depth = depth + 1

            flat_size = action_size * batch_size
            flat_states = xnp.reshape(neighbours, (flat_size,))
            flat_g = child_g.reshape((flat_size,))
            flat_depth = jnp.tile(child_depth, (action_size,)).reshape((flat_size,))

            flat_parent_valid = jnp.tile(valid, (action_size,))
            flat_valid = jnp.logical_and(flat_parent_valid, jnp.isfinite(flat_g))

            is_solved_mask = puzzle.batched_is_solved(solve_config, flat_states)
            is_solved_mask = jnp.logical_and(is_solved_mask, flat_valid)
            any_solved = jnp.any(is_solved_mask)

            first_idx = jnp.argmax(is_solved_mask)
            found_sol_state = xnp.expand_dims(flat_states[first_idx], 0)
            found_sol_cost = flat_g[first_idx]

            new_solved = jnp.logical_or(frontier.solved, any_solved)
            new_sol_state = jax.lax.cond(
                any_solved, lambda _: found_sol_state, lambda _: frontier.solution_state, None
            )
            new_sol_cost = jax.lax.cond(
                any_solved, lambda _: found_sol_cost, lambda _: frontier.solution_cost, None
            )

            q_vals = variable_q_parent_switcher(q_params, states, valid).astype(KEY_DTYPE)
            q_vals = jnp.where(valid[:, None], q_vals, jnp.inf)
            q_vals = q_vals.transpose()
            f_vals = (cost_weight * gs[jnp.newaxis, :] + q_vals).astype(KEY_DTYPE)
            flat_f = f_vals.reshape((flat_size,))

            sort_keys_pre = jnp.where(flat_valid, 0, 1)
            perm_pre = jnp.argsort(sort_keys_pre)

            states_pre = flat_states[perm_pre]
            gs_pre = flat_g[perm_pre]
            depths_pre = flat_depth[perm_pre]
            valid_pre = flat_valid[perm_pre]
            fs_pre = flat_f[perm_pre]

            unique_mask_pre = xnp.unique_mask(states_pre, key=gs_pre, filled=valid_pre)
            valid_after_dedup = jnp.logical_and(valid_pre, unique_mask_pre)

            f_safe = jnp.nan_to_num(fs_pre, nan=1e5, posinf=1e5)
            sort_keys_final = jnp.where(valid_after_dedup, f_safe, 1e6)
            perm_final = jnp.argsort(sort_keys_final)

            top_indices = perm_final[:batch_size]

            new_states = states_pre[top_indices]
            new_costs = gs_pre[top_indices]
            new_depths = depths_pre[top_indices]
            new_valid = valid_after_dedup[top_indices]

            new_frontier = IDFrontier(
                states=new_states,
                costs=new_costs,
                depths=new_depths,
                valid_mask=new_valid,
                solved=new_solved,
                solution_state=new_sol_state,
                solution_cost=new_sol_cost,
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
):
    statecls = puzzle.State
    action_size = puzzle.action_size

    variable_q_parent_switcher = variable_batch_switcher_builder(
        q_fn.batched_q_value,
        max_batch_size=batch_size,
        min_batch_size=MIN_BATCH_SIZE,
        pad_value=jnp.inf,
    )

    def _min_q(q_params, states, valid_mask):
        q_vals = variable_q_parent_switcher(q_params, states, valid_mask).astype(KEY_DTYPE)
        q_vals = jnp.where(valid_mask[:, None], q_vals, jnp.inf)
        return jnp.min(q_vals, axis=-1)

    generate_frontier = _id_qstar_frontier_builder(
        puzzle,
        q_fn,
        batch_size=batch_size,
        cost_weight=cost_weight,
    )

    def _push_frontier_to_stack(sr, frontier, q_params, bound):
        hs = _min_q(q_params, frontier.states, frontier.valid_mask)
        fs = (cost_weight * frontier.costs + hs).astype(KEY_DTYPE)

        keep_mask = jnp.logical_and(frontier.valid_mask, fs <= bound + 1e-6)
        prune_mask = jnp.logical_and(frontier.valid_mask, fs > bound + 1e-6)
        pruned_fs = jnp.where(prune_mask, fs, jnp.inf)
        min_pruned = jnp.min(pruned_fs).astype(KEY_DTYPE)

        new_next_bound = jnp.minimum(sr.next_bound, min_pruned).astype(KEY_DTYPE)
        sr = sr.replace(next_bound=new_next_bound)

        actions = jnp.full((batch_size,), -1, dtype=jnp.int32)

        return sr.push_batch(frontier.states, frontier.costs, frontier.depths, actions, keep_mask)

    def init_loop_state(solve_config: Puzzle.SolveConfig, start: Puzzle.State, **kwargs):
        search_result = IDSearchResult.build(
            statecls,
            capacity=max_nodes,
            action_size=action_size,
        )

        q_parameters = q_fn.prepare_q_parameters(solve_config, **kwargs)

        frontier = generate_frontier(
            solve_config,
            start,
            h_params=q_parameters,
        )

        frontier_h = _min_q(q_parameters, frontier.states, frontier.valid_mask).astype(KEY_DTYPE)
        frontier_f = (cost_weight * frontier.costs + frontier_h).astype(KEY_DTYPE)

        valid_fs = jnp.where(frontier.valid_mask, frontier_f, jnp.inf)
        start_bound = jnp.min(valid_fs).astype(KEY_DTYPE)

        search_result = search_result.replace(
            bound=start_bound,
            next_bound=jnp.array(jnp.inf, dtype=KEY_DTYPE),
            solved=frontier.solved,
            solution_state=frontier.solution_state,
            solution_cost=frontier.solution_cost,
            solved_idx=jnp.where(frontier.solved, 0, -1),
        )

        search_result = _push_frontier_to_stack(search_result, frontier, q_parameters, start_bound)

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

        sr, parents, parent_costs, parent_depths, valid_mask, _ = sr.get_top_batch(batch_size)

        is_solved_mask = puzzle.batched_is_solved(solve_config, parents)
        is_solved_mask = jnp.logical_and(is_solved_mask, valid_mask)
        any_solved = jnp.any(is_solved_mask)

        first_idx = jnp.argmax(is_solved_mask)
        new_sol_state = xnp.expand_dims(parents[first_idx], 0)
        new_sol_cost = parent_costs[first_idx]

        sr_solved = sr.replace(
            solved=jnp.logical_or(sr.solved, any_solved),
            solved_idx=jnp.where(any_solved, 0, -1),
            solution_state=jax.lax.cond(
                any_solved, lambda _: new_sol_state, lambda _: sr.solution_state, None
            ),
            solution_cost=jax.lax.cond(
                any_solved,
                lambda _: new_sol_cost.astype(KEY_DTYPE),
                lambda _: sr.solution_cost,
                None,
            ),
        )

        neighbours, step_costs = puzzle.batched_get_neighbours(solve_config, parents, valid_mask)
        child_costs = parent_costs[jnp.newaxis, :] + step_costs
        child_depths = parent_depths + 1

        flat_size = action_size * batch_size
        flat_neighbours = xnp.reshape(neighbours, (flat_size,))
        flat_g = child_costs.reshape((flat_size,))
        flat_depth = jnp.tile(child_depths, (action_size,)).reshape((flat_size,))

        flat_valid_parent = jnp.tile(valid_mask, (action_size,))
        flat_valid = jnp.logical_and(flat_valid_parent, jnp.isfinite(flat_g))

        flat_actions = jnp.tile(
            jnp.arange(action_size, dtype=jnp.int32)[:, None], (1, batch_size)
        ).reshape((flat_size,))

        unique_mask = xnp.unique_mask(flat_neighbours, key=flat_g, filled=flat_valid)
        flat_valid = jnp.logical_and(flat_valid, unique_mask)

        q_vals = variable_q_parent_switcher(params, parents, valid_mask).astype(KEY_DTYPE)
        q_vals = jnp.where(valid_mask[:, None], q_vals, jnp.inf)
        q_vals = q_vals.transpose()
        f_vals = (cost_weight * parent_costs[jnp.newaxis, :] + q_vals).astype(KEY_DTYPE)
        flat_f = f_vals.reshape((flat_size,))

        return_sr = jax.lax.cond(
            any_solved,
            lambda s: s,
            lambda s: _expand_step(
                s,
                flat_neighbours,
                flat_g,
                flat_depth,
                flat_actions,
                flat_valid,
                flat_f,
            ),
            sr_solved,
        )

        return loop_state.replace(search_result=return_sr)

    def _expand_step(sr, states, gs, depths, actions, valid, fs):
        sort_keys = jnp.where(valid, 0, 1)
        perm = jnp.argsort(sort_keys)

        states_sorted = states[perm]
        gs_sorted = gs[perm]
        depths_sorted = depths[perm]
        actions_sorted = actions[perm]
        valid_sorted = valid[perm]
        fs_sorted = fs[perm]

        active_bound = sr.bound
        keep_mask_sorted = jnp.logical_and(valid_sorted, fs_sorted <= active_bound + 1e-6)
        prune_mask_sorted = jnp.logical_and(valid_sorted, fs_sorted > active_bound + 1e-6)

        pruned_fs = jnp.where(prune_mask_sorted, fs_sorted, jnp.inf)
        min_pruned = jnp.min(pruned_fs).astype(KEY_DTYPE)

        new_next_bound = jnp.minimum(sr.next_bound, min_pruned).astype(KEY_DTYPE)
        sr = sr.replace(next_bound=new_next_bound)

        # Optimization: Sort valid children by f-value descending
        # (Worst -> Best) such that valid nodes are first, then invalid.
        f_key = jnp.where(keep_mask_sorted, -fs_sorted, jnp.inf)
        perm_f = jnp.argsort(f_key)

        states_ordered = states_sorted[perm_f]
        gs_ordered = gs_sorted[perm_f]
        depths_ordered = depths_sorted[perm_f]
        actions_ordered = actions_sorted[perm_f]
        keep_mask_ordered = keep_mask_sorted[perm_f]

        return sr.push_batch(
            states_ordered,
            gs_ordered,
            depths_ordered,
            actions_ordered,
            keep_mask_ordered,
        )

    def outer_cond(loop_state: IDLoopState):
        sr = loop_state.search_result
        return jnp.logical_and(~sr.solved, jnp.isfinite(sr.bound))

    def outer_body(loop_state: IDLoopState):
        loop_state = jax.lax.while_loop(inner_cond, inner_body, loop_state)

        sr = loop_state.search_result
        new_bound = sr.next_bound

        reset_sr = sr.replace(
            bound=new_bound,
            next_bound=jnp.array(jnp.inf, dtype=KEY_DTYPE),
            stack=sr.stack.replace(size=jnp.array(0, dtype=jnp.uint32)),
        )

        reset_sr = _push_frontier_to_stack(
            reset_sr, loop_state.frontier, loop_state.params, new_bound
        )

        return loop_state.replace(search_result=reset_sr)

    return init_loop_state, outer_cond, outer_body


def id_qstar_builder(
    puzzle: Puzzle,
    q_fn: QFunction,
    batch_size: int = 1024,
    max_nodes: int = int(1e6),
    cost_weight: float = 1.0,
    pop_ratio: float = 1.0,
    show_compile_time: bool = False,
):
    init_loop, cond, body = _id_qstar_loop_builder(puzzle, q_fn, batch_size, max_nodes, cost_weight)

    def id_qstar(solve_config: Puzzle.SolveConfig, start: Puzzle.State, **kwargs):
        loop_state = init_loop(solve_config, start, **kwargs)
        loop_state = jax.lax.while_loop(cond, body, loop_state)
        return loop_state.search_result

    id_qstar_fn = jax.jit(id_qstar)
    empty_solve_config = puzzle.SolveConfig.default()
    empty_states = puzzle.State.default()

    if show_compile_time:
        print("initializing jit for ID-Q*")
        start_t = time.time()
    id_qstar_fn(empty_solve_config, empty_states)
    if show_compile_time:
        print(f"JIT compile time: {time.time() - start_t:.2f}s")
        print("JIT compiled\n\n")

    return id_qstar_fn
