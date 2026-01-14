"""
Iterative Deepening Q* (ID-Q*) Search Implementation
"""

import time

import jax
import jax.numpy as jnp
import xtructure.numpy as xnp
from puxle import Puzzle

from JAxtar.annotate import KEY_DTYPE
from JAxtar.id_stars.search_base import IDLoopState, IDSearchResult
from qfunction.q_base import QFunction


def _id_qstar_loop_builder(
    puzzle: Puzzle,
    q_fn: QFunction,
    batch_size: int = 1024,
    max_nodes: int = int(1e6),
    cost_weight: float = 1.0,
):
    statecls = puzzle.State
    action_size = puzzle.action_size

    # Direct min Q-value function (no batch switcher needed for ID*)
    def min_q_fn(q_params, states, valid=None):
        q_values = q_fn.batched_q_value(q_params, states)  # [batch, action_size]
        return jnp.min(q_values, axis=1)  # [batch]

    def init_loop_state(solve_config: Puzzle.SolveConfig, start: Puzzle.State, **kwargs):
        # Initialize Result
        search_result = IDSearchResult.build(
            statecls,
            capacity=max_nodes,
            action_size=action_size,
        )

        q_parameters = q_fn.prepare_q_parameters(solve_config, **kwargs)

        # Initial Q-values
        # For the start node, f(start) = min_a ( Q(start, a) )  (since g=0)
        # OR usually IDA* bound is on f-value.
        # ID-Q* typically uses f = g + min_a Q(s,a)?
        # Or if Q(s,a) estimates total cost, then f = Q(s,a)?
        # Usually Q(s,a) includes cost to go, so f = g + min(Q).

        # q_values: [actions, batch]
        start_reshaped = xnp.expand_dims(start, axis=0)
        start_q = q_fn.batched_q_value(
            q_parameters,
            start_reshaped,
        )

        # Min Q value across actions represents the best estimated cost from start
        min_start_q = jnp.min(start_q)
        start_cost = 0.0
        start_f = (cost_weight * start_cost + min_start_q).astype(KEY_DTYPE)

        # Initialize Bound
        search_result = search_result.replace(
            bound=start_f,
            next_bound=jnp.array(jnp.inf, dtype=KEY_DTYPE),
        )

        def _push_start(sr):
            valid_mask = jnp.zeros((batch_size,), dtype=jnp.bool_).at[0].set(True)
            states_batch = xnp.pad(start, (0, batch_size - 1))
            costs_batch = jnp.zeros((batch_size,), dtype=KEY_DTYPE)
            depths_batch = jnp.zeros((batch_size,), dtype=jnp.int32)
            actions_batch = jnp.full((batch_size,), -1, dtype=jnp.int32)
            return sr.push_batch(states_batch, costs_batch, depths_batch, actions_batch, valid_mask)

        search_result = _push_start(search_result)

        return IDLoopState(
            search_result=search_result,
            solve_config=solve_config,
            params=q_parameters,
            start_state=start,
            start_cost=jnp.array(start_cost, dtype=KEY_DTYPE),
        )

    # -----------------------------------------------------------------------
    # INNER LOOP
    # -----------------------------------------------------------------------
    def inner_cond(loop_state: IDLoopState):
        sr = loop_state.search_result
        has_items = sr.stack_ptr > 0
        not_solved = ~sr.solved
        return jnp.logical_and(has_items, not_solved)

    def inner_body(loop_state: IDLoopState):
        sr = loop_state.search_result
        solve_config = loop_state.solve_config
        params = loop_state.params

        sr, parents, parent_costs, parent_depths, valid_mask, _ = sr.get_top_batch(batch_size)

        is_solved_mask = puzzle.batched_is_solved(solve_config, parents)
        is_solved_mask = jnp.logical_and(is_solved_mask, valid_mask)
        any_solved = jnp.any(is_solved_mask)

        # Capture solution
        first_solved_idx = jnp.argmax(is_solved_mask)
        solved_st = parents[first_solved_idx]
        solved_cost = parent_costs[first_solved_idx]
        solved_st_batch = xnp.expand_dims(solved_st, axis=0)

        new_solution_state = jax.lax.cond(
            any_solved, lambda _: solved_st_batch, lambda _: sr.solution_state, None
        )
        new_solution_cost = jax.lax.cond(
            any_solved, lambda _: solved_cost.astype(KEY_DTYPE), lambda _: sr.solution_cost, None
        )

        sr_solved = sr.replace(
            solved=jnp.logical_or(sr.solved, any_solved),
            solved_idx=jnp.where(any_solved, 0, -1),
            solution_state=new_solution_state,
            solution_cost=new_solution_cost,
        )

        # Expand neighbours
        neighbours, step_costs = puzzle.batched_get_neighbours(solve_config, parents, valid_mask)

        # neighbours: [action_size, batch_size] Xtructurable
        # step_costs: [action_size, batch_size] array
        child_costs = parent_costs[jnp.newaxis, :] + step_costs  # [action_size, batch_size]
        child_depths = parent_depths + 1  # [batch_size]

        flat_size = action_size * batch_size
        flat_neighbours = xnp.reshape(neighbours, (flat_size,))
        flat_g = child_costs.reshape((flat_size,))
        flat_depth = jnp.tile(child_depths, (action_size,)).reshape((flat_size,))

        actions_matrix = jnp.tile(
            jnp.arange(action_size, dtype=jnp.int32)[:, None], (1, batch_size)
        )
        flat_actions = actions_matrix.reshape((flat_size,))

        flat_valid_parent = jnp.tile(valid_mask, (action_size,))
        flat_valid = jnp.logical_and(flat_valid_parent, jnp.isfinite(flat_g))

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
                params,
                min_q_fn,
            ),
            sr_solved,
        )

        return loop_state.replace(search_result=return_sr)

    def _expand_step(sr, states, gs, depths, actions, valid, q_params, q_fn):
        # Compact valid items to the front for efficient Q evaluation
        sort_keys = jnp.where(valid, 0, 1)  # 0 for valid, 1 for invalid
        perm = jnp.argsort(sort_keys)

        states_sorted = states[perm]
        gs_sorted = gs[perm]
        depths_sorted = depths[perm]
        actions_sorted = actions[perm]
        valid_sorted = valid[perm]  # Now True values are at the front

        # Call Q-function wrapper on sorted (compacted) states
        # q_fn is the min_q wrapper via batch_switcher, returns [flat_size]
        hs_sorted = q_fn(q_params, states_sorted, valid_sorted).astype(KEY_DTYPE)
        fs_sorted = (cost_weight * gs_sorted + hs_sorted).astype(KEY_DTYPE)

        active_bound = sr.bound

        # Prune check (on sorted data)
        keep_mask_sorted = jnp.logical_and(valid_sorted, fs_sorted <= active_bound + 1e-6)
        prune_mask_sorted = jnp.logical_and(valid_sorted, fs_sorted > active_bound + 1e-6)

        pruned_fs = jnp.where(prune_mask_sorted, fs_sorted, jnp.array(jnp.inf, dtype=KEY_DTYPE))
        min_pruned_f = jnp.min(pruned_fs).astype(KEY_DTYPE)
        new_next_bound = jnp.minimum(sr.next_bound, min_pruned_f).astype(KEY_DTYPE)

        sr_next = sr.replace(next_bound=new_next_bound)

        # Push kept nodes (states are already sorted)
        sr_final = sr_next.push_batch(
            states_sorted, gs_sorted, depths_sorted, actions_sorted, keep_mask_sorted
        )
        return sr_final

    # -----------------------------------------------------------------------
    # OUTER LOOP
    # -----------------------------------------------------------------------
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
            stack_ptr=jnp.array(0, dtype=jnp.int32),
        )

        def _push_start(sr, start):
            valid_mask = jnp.zeros((batch_size,), dtype=jnp.bool_).at[0].set(True)
            states_batch = xnp.pad(start, (0, batch_size - 1))
            costs_batch = jnp.zeros((batch_size,), dtype=KEY_DTYPE)
            depths_batch = jnp.zeros((batch_size,), dtype=jnp.int32)
            actions_batch = jnp.full((batch_size,), -1, dtype=jnp.int32)
            return sr.push_batch(states_batch, costs_batch, depths_batch, actions_batch, valid_mask)

        reset_sr = _push_start(reset_sr, loop_state.start_state)
        return loop_state.replace(search_result=reset_sr)

    return init_loop_state, outer_cond, outer_body


def id_qstar_builder(
    puzzle: Puzzle,
    q_fn: QFunction,
    batch_size: int = 1024,
    max_nodes: int = int(1e6),
    cost_weight: float = 1.0,
    pop_ratio: float = 1.0,  # Unused
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
        print("initializing jit")
        start_time = time.time()

    id_qstar_fn(empty_solve_config, empty_states)

    if show_compile_time:
        end_time = time.time()
        print(f"Compile Time: {end_time - start_time:6.2f} seconds")
        print("JIT compiled\n\n")

    return id_qstar_fn
