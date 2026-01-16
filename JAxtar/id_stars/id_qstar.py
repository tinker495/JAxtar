"""
Iterative Deepening Q* (ID-Q*) Search Implementation
"""

import time

import jax
import jax.numpy as jnp
import xtructure.numpy as xnp
from puxle import Puzzle
from xtructure import FieldDescriptor, xtructure_dataclass

from JAxtar.annotate import KEY_DTYPE, MIN_BATCH_SIZE
from JAxtar.id_stars.search_base import IDFrontier, IDLoopState, IDSearchResult
from JAxtar.id_stars.utils import _apply_non_backtracking
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

    @xtructure_dataclass
    class FrontierFlatBatch:
        state: FieldDescriptor.scalar(dtype=statecls)
        cost: FieldDescriptor.scalar(dtype=KEY_DTYPE)
        depth: FieldDescriptor.scalar(dtype=jnp.int32)
        trail: FieldDescriptor.tensor(dtype=statecls, shape=(non_backtracking_steps,))
        action_history: FieldDescriptor.tensor(dtype=jnp.int32, shape=(max_path_len,))

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
        trail_padded = statecls.default((batch_size, non_backtracking_steps))
        action_history_padded = jnp.full((batch_size, max_path_len), -1, dtype=jnp.int32)
        action_ids = jnp.arange(action_size, dtype=jnp.int32)

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
        solution_actions = jnp.full((max_path_len,), -1, dtype=jnp.int32)

        init_val = IDFrontier(
            states=start_padded,
            costs=costs_padded,
            depths=depths_padded,
            valid_mask=valid_padded,
            f_scores=costs_padded,
            trail=trail_padded,
            solved=root_solved,
            solution_state=solution_state,
            solution_cost=solution_cost,
            solution_actions_arr=solution_actions,
            action_history=action_history_padded,
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
            trail = frontier.trail
            action_history = frontier.action_history

            neighbours, step_costs = puzzle.batched_get_neighbours(solve_config, states, valid)

            child_g = (gs[jnp.newaxis, :] + step_costs).astype(KEY_DTYPE)
            child_depth = depth + 1
            if non_backtracking_steps > 0:
                parent_trail_tiled = xnp.stack([trail] * action_size, axis=0)
                parent_states_tiled = xnp.stack([states] * action_size, axis=0)
                parent_state_exp = xnp.expand_dims(parent_states_tiled, axis=2)
                shifted_trail = parent_trail_tiled[:, :, :-1]
                child_trail = xnp.concatenate((parent_state_exp, shifted_trail), axis=2)
                flat_trail = xnp.reshape(child_trail, (flat_size, non_backtracking_steps))
            else:
                flat_trail = empty_trail_flat

            flat_action_history = jnp.broadcast_to(
                action_history[:, None, :], (batch_size, action_size, max_path_len)
            )
            flat_action_history = flat_action_history.reshape((flat_size, max_path_len))
            flat_actions = jnp.tile(action_ids, batch_size)

            # Update history: insert action at current depth
            flat_depth_int = depth.astype(jnp.int32)  # batch_size
            flat_depth_tiled = jnp.broadcast_to(flat_depth_int[:, None], (batch_size, action_size))
            flat_depth_flat = flat_depth_tiled.reshape((flat_size,))

            flat_action_history = flat_action_history.at[
                jnp.arange(flat_size), flat_depth_flat
            ].set(flat_actions)

            flat_states = xnp.reshape(neighbours, (flat_size,))
            flat_g = child_g.reshape((flat_size,))
            flat_depth = jnp.broadcast_to(child_depth, (action_size, batch_size)).reshape(
                (flat_size,)
            )

            flat_parent_valid = jnp.broadcast_to(valid, (action_size, batch_size)).reshape(
                (flat_size,)
            )
            flat_valid = jnp.logical_and(flat_parent_valid, jnp.isfinite(flat_g))

            is_solved_mask = puzzle.batched_is_solved(solve_config, flat_states)
            is_solved_mask = jnp.logical_and(is_solved_mask, flat_valid)
            any_solved = jnp.any(is_solved_mask)

            first_idx = jnp.argmax(is_solved_mask)
            found_sol_state = xnp.expand_dims(flat_states[first_idx], 0)
            found_sol_cost = flat_g[first_idx]
            found_sol_actions = flat_action_history[first_idx]

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
            q_vals = q_vals.transpose()
            f_vals = (cost_weight * gs[jnp.newaxis, :] + q_vals).astype(KEY_DTYPE)
            flat_f = f_vals.reshape((flat_size,))

            unique_mask, unique_idx = xnp.unique_mask(
                flat_states,
                key=flat_g,
                filled=flat_valid,
                return_index=True,
            )
            unique_count = jnp.sum(unique_mask.astype(jnp.int32))
            flat_valid = jnp.logical_and(flat_valid, unique_mask)
            flat_valid = _apply_non_backtracking(
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

            flat_batch = FrontierFlatBatch(
                state=flat_states,
                cost=flat_g,
                depth=flat_depth,
                trail=flat_trail,
                action_history=flat_action_history,
            )
            packed_batch = xnp.take(flat_batch, unique_idx, axis=0)
            packed_f = xnp.take(flat_f, unique_idx, axis=0)
            packed_valid = flat_valid[unique_idx]
            packed_positions = jnp.arange(flat_size, dtype=jnp.int32) < unique_count
            packed_valid = jnp.logical_and(packed_valid, packed_positions)

            f_safe = jnp.where(packed_valid, jnp.nan_to_num(packed_f, nan=1e5, posinf=1e5), jnp.inf)
            neg_f = -f_safe
            top_vals, top_indices = jax.lax.top_k(neg_f, batch_size)
            selected_f = -top_vals
            selected_valid = jnp.isfinite(selected_f)

            selected = xnp.take(packed_batch, top_indices, axis=0)
            new_valid = jnp.logical_and(selected_valid, packed_valid[top_indices])

            new_frontier = IDFrontier(
                states=selected.state,
                costs=selected.cost,
                depths=selected.depth,
                valid_mask=new_valid,
                f_scores=selected_f,
                trail=selected.trail,
                solved=new_solved,
                solution_state=new_sol_state,
                solution_cost=new_sol_cost,
                solution_actions_arr=new_sol_actions,
                action_history=selected.action_history,
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
    flat_actions = jnp.broadcast_to(action_ids[:, None], (action_size, batch_size)).reshape(
        (flat_size,)
    )
    frontier_actions = jnp.full((batch_size,), -1, dtype=jnp.int32)

    @xtructure_dataclass
    class ExpandFlatBatch:
        state: FieldDescriptor.scalar(dtype=statecls)
        cost: FieldDescriptor.scalar(dtype=KEY_DTYPE)
        depth: FieldDescriptor.scalar(dtype=jnp.int32)
        action: FieldDescriptor.scalar(dtype=jnp.int32)
        trail: FieldDescriptor.tensor(dtype=statecls, shape=(non_backtracking_steps,))
        action_history: FieldDescriptor.tensor(dtype=jnp.int32, shape=(max_path_len,))

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
        non_backtracking_steps=non_backtracking_steps,
        max_path_len=max_path_len,
    )

    def _push_frontier_to_stack(sr, frontier, bound):
        fs = frontier.f_scores

        keep_mask = jnp.logical_and(frontier.valid_mask, fs <= bound + 1e-6)
        prune_mask = jnp.logical_and(frontier.valid_mask, fs > bound + 1e-6)
        pruned_fs = jnp.where(prune_mask, fs, jnp.inf)
        min_pruned = jnp.min(pruned_fs).astype(KEY_DTYPE)

        new_next_bound = jnp.minimum(sr.next_bound, min_pruned).astype(KEY_DTYPE)
        sr = sr.replace(next_bound=new_next_bound)

        return sr.push_batch(
            frontier.states,
            frontier.costs,
            frontier.depths,
            frontier_actions,
            frontier.trail,
            frontier.action_history,
            keep_mask,
        )

    def init_loop_state(solve_config: Puzzle.SolveConfig, start: Puzzle.State, **kwargs):
        search_result = IDSearchResult.build(
            statecls,
            capacity=max_nodes,
            action_size=action_size,
            non_backtracking_steps=non_backtracking_steps,
            max_path_len=max_path_len,
        )

        q_parameters = q_fn.prepare_q_parameters(solve_config, **kwargs)

        frontier = generate_frontier(solve_config, start, h_params=q_parameters)

        frontier_h = _min_q(q_parameters, frontier.states, frontier.valid_mask).astype(KEY_DTYPE)
        frontier_f = (cost_weight * frontier.costs + frontier_h).astype(KEY_DTYPE)
        frontier = frontier.replace(f_scores=frontier_f)

        valid_fs = jnp.where(frontier.valid_mask, frontier.f_scores, jnp.inf)
        start_bound = jnp.min(valid_fs).astype(KEY_DTYPE)

        search_result = search_result.replace(
            bound=start_bound,
            next_bound=jnp.array(jnp.inf, dtype=KEY_DTYPE),
            solved=frontier.solved,
            solution_state=frontier.solution_state,
            solution_cost=frontier.solution_cost,
            solution_actions_arr=frontier.solution_actions_arr,
            solved_idx=jnp.where(frontier.solved, 0, -1),
        )

        search_result = _push_frontier_to_stack(search_result, frontier, start_bound)

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
            parents,
            parent_costs,
            parent_depths,
            parent_trails,
            parent_action_histories,
            valid_mask,
            _,
        ) = sr.get_top_batch(batch_size)

        is_solved_mask = puzzle.batched_is_solved(solve_config, parents)
        is_solved_mask = jnp.logical_and(is_solved_mask, valid_mask)
        any_solved = jnp.any(is_solved_mask)

        first_idx = jnp.argmax(is_solved_mask)
        new_sol_state = xnp.expand_dims(parents[first_idx], 0)
        new_sol_cost = parent_costs[first_idx]
        new_sol_actions = parent_action_histories[first_idx]

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
            solution_actions_arr=jax.lax.cond(
                any_solved,
                lambda _: new_sol_actions,
                lambda _: sr.solution_actions_arr,
                None,
            ),
        )

        neighbours, step_costs = puzzle.batched_get_neighbours(solve_config, parents, valid_mask)
        child_costs = parent_costs[jnp.newaxis, :] + step_costs
        child_depths = parent_depths + 1
        if non_backtracking_steps > 0:
            parent_trail_tiled = xnp.stack([parent_trails] * action_size, axis=0)
            parent_states_tiled = xnp.stack([parents] * action_size, axis=0)
            parent_state_exp = xnp.expand_dims(parent_states_tiled, axis=2)
            shifted_trail = parent_trail_tiled[:, :, :-1]
            child_trail = xnp.concatenate((parent_state_exp, shifted_trail), axis=2)
            flat_trail = xnp.reshape(child_trail, (flat_size, non_backtracking_steps))
        else:
            flat_trail = empty_trail_flat

        flat_action_history = jnp.broadcast_to(
            parent_action_histories[:, None, :], (batch_size, action_size, max_path_len)
        )
        flat_action_history = flat_action_history.reshape((flat_size, max_path_len))

        flat_depth_int = parent_depths.astype(jnp.int32)
        flat_depth_tiled = jnp.broadcast_to(
            flat_depth_int[:, None], (batch_size, action_size)
        )  # [batch, action]
        flat_depth_flat = flat_depth_tiled.reshape((flat_size,))

        flat_action_history = flat_action_history.at[jnp.arange(flat_size), flat_depth_flat].set(
            flat_actions
        )

        flat_neighbours = xnp.reshape(neighbours, (flat_size,))
        flat_g = child_costs.reshape((flat_size,))
        flat_depth = jnp.broadcast_to(child_depths, (action_size, batch_size)).reshape((flat_size,))

        flat_valid_parent = jnp.broadcast_to(valid_mask, (action_size, batch_size)).reshape(
            (flat_size,)
        )
        flat_valid = jnp.logical_and(flat_valid_parent, jnp.isfinite(flat_g))
        flat_valid = jnp.logical_and(flat_valid, flat_depth <= max_path_len)

        unique_mask = xnp.unique_mask(flat_neighbours, key=flat_g, filled=flat_valid)
        flat_valid = jnp.logical_and(flat_valid, unique_mask)
        flat_valid = _apply_non_backtracking(
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
                flat_trail,
                flat_action_history,
                flat_valid,
                flat_f,
            ),
            sr_solved,
        )

        return loop_state.replace(search_result=return_sr)

    def _expand_step(sr, states, gs, depths, actions, trails, action_histories, valid, fs):
        active_bound = sr.bound
        keep_mask = jnp.logical_and(valid, fs <= active_bound + 1e-6)

        # Optimization: Sort valid children by f-value descending (Worst -> Best)
        # Key: -fs for kept items, inf for others (to push to end)
        f_key = jnp.where(keep_mask, -fs, jnp.inf)
        perm = jnp.argsort(f_key)

        flat_batch = ExpandFlatBatch(
            state=states,
            cost=gs,
            depth=depths,
            action=actions,
            trail=trails,
            action_history=action_histories,
        )
        ordered = xnp.take(flat_batch, perm, axis=0)

        n_push = jnp.sum(keep_mask)

        # Update next bound
        prune_mask = jnp.logical_and(valid, fs > active_bound + 1e-6)
        pruned_fs = jnp.where(prune_mask, fs, jnp.inf)
        min_pruned = jnp.min(pruned_fs).astype(KEY_DTYPE)
        new_next_bound = jnp.minimum(sr.next_bound, min_pruned).astype(KEY_DTYPE)
        sr = sr.replace(next_bound=new_next_bound)

        return sr.push_packed_batch(
            ordered.state,
            ordered.cost,
            ordered.depth,
            ordered.action,
            ordered.trail,
            ordered.action_history,
            n_push,
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

        reset_sr = _push_frontier_to_stack(reset_sr, loop_state.frontier, new_bound)

        return loop_state.replace(search_result=reset_sr)

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
