"""
Iterative Deepening Q* (ID-Q*) Search Implementation
"""

import time

import jax
import jax.numpy as jnp
import xtructure.numpy as xnp
from puxle import Puzzle

from JAxtar.annotate import KEY_DTYPE, MIN_BATCH_SIZE
from JAxtar.id_stars.frontier import frontier_builder
from JAxtar.id_stars.search_base import IDLoopState, IDSearchResult
from JAxtar.utils.batch_switcher import variable_batch_switcher_builder
from qfunction.q_base import QFunction


class QMinHeuristic:
    """Wrapper to make QFunction look like a Heuristic (min Q) for Frontier generation."""

    def __init__(self, q_fn):
        self.q_fn = q_fn

    def batched_distance(self, params, states, **kwargs):
        # Q-values: [batch, actions]
        q_vals = self.q_fn.batched_q_value(params, states)
        # We use min_a Q(s,a) as the 'heuristic h(s)'
        return jnp.min(q_vals, axis=-1)


def _id_qstar_loop_builder(
    puzzle: Puzzle,
    q_fn: QFunction,
    batch_size: int = 1024,
    max_nodes: int = int(1e6),
    cost_weight: float = 1.0,
):
    statecls = puzzle.State
    action_size = puzzle.action_size

    # 1. Builders for Switchers
    # Q-values: [batch, action_size]
    # We want min_q: [batch]
    def _base_min_q(params, states):
        q_values = q_fn.batched_q_value(params, states)
        return jnp.min(q_values, axis=1)

    # Search step switcher (for expanded nodes: size = batch * action)
    variable_q_step_switcher = variable_batch_switcher_builder(
        _base_min_q,
        max_batch_size=batch_size * action_size,
        min_batch_size=MIN_BATCH_SIZE,
        pad_value=jnp.inf,
    )

    # Frontier switcher (size = batch_size)
    variable_q_frontier_switcher = variable_batch_switcher_builder(
        _base_min_q,
        max_batch_size=batch_size,
        min_batch_size=MIN_BATCH_SIZE,
        pad_value=jnp.inf,
    )

    # Min Q wrapper for frontier builder
    q_heuristic_wrapper = QMinHeuristic(q_fn)

    # 2. Shared Frontier Helper
    def _push_frontier_to_stack(sr, frontier, q_params, bound):
        # Calc F = g + min(Q(s))
        hs = variable_q_frontier_switcher(q_params, frontier.states, frontier.valid_mask).astype(
            KEY_DTYPE
        )
        fs = (cost_weight * frontier.costs + hs).astype(KEY_DTYPE)

        # Determine what to push
        keep_mask = jnp.logical_and(frontier.valid_mask, fs <= bound + 1e-6)

        # Determine what to start next_bound with
        prune_mask = jnp.logical_and(frontier.valid_mask, fs > bound + 1e-6)
        pruned_fs = jnp.where(prune_mask, fs, jnp.inf)
        min_pruned = jnp.min(pruned_fs).astype(KEY_DTYPE)

        new_next_bound = jnp.minimum(sr.next_bound, min_pruned).astype(KEY_DTYPE)
        sr = sr.replace(next_bound=new_next_bound)

        # Actions for frontier nodes? Default -1
        actions = jnp.full((batch_size,), -1, dtype=jnp.int32)

        return sr.push_batch(frontier.states, frontier.costs, frontier.depths, actions, keep_mask)

    # 3. Inner Loop Definitions
    def init_loop_state(solve_config: Puzzle.SolveConfig, start: Puzzle.State, **kwargs):
        # Initialize Result
        search_result = IDSearchResult.build(
            statecls,
            capacity=max_nodes,
            action_size=action_size,
        )

        q_parameters = q_fn.prepare_q_parameters(solve_config, **kwargs)

        # 1. Generate Frontier with Q-based wrapper
        generate_frontier_fn = frontier_builder(puzzle, q_heuristic_wrapper, batch_size=batch_size)

        frontier = generate_frontier_fn(
            solve_config,
            start,
            h_params=q_parameters,
        )

        # 2. Initial Bound Setup
        # Use direct call for single root f-value if needed? No, we use frontier.
        # But we need h for ALL nodes in frontier to find min.
        frontier_h = variable_q_frontier_switcher(
            q_parameters, frontier.states, frontier.valid_mask
        ).astype(KEY_DTYPE)
        frontier_f = (cost_weight * frontier.costs + frontier_h).astype(KEY_DTYPE)

        valid_fs = jnp.where(frontier.valid_mask, frontier_f, jnp.inf)
        start_bound = jnp.min(valid_fs).astype(KEY_DTYPE)

        # Initialize IDSearchResult with this bound
        search_result = search_result.replace(
            bound=start_bound,
            next_bound=jnp.array(jnp.inf, dtype=KEY_DTYPE),
            solved=frontier.solved,
            solution_state=frontier.solution_state,
            solution_cost=frontier.solution_cost,
            solved_idx=jnp.where(frontier.solved, 0, -1),
        )

        # 3. Push Initial Frontier
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

        # 1. Pop
        sr, parents, parent_costs, parent_depths, valid_mask, _ = sr.get_top_batch(batch_size)

        # 2. Check Solved
        is_solved_mask = puzzle.batched_is_solved(solve_config, parents)
        is_solved_mask = jnp.logical_and(is_solved_mask, valid_mask)
        any_solved = jnp.any(is_solved_mask)

        # Solution capture
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

        # 3. Expansion
        neighbours, step_costs = puzzle.batched_get_neighbours(solve_config, parents, valid_mask)
        child_costs = parent_costs[jnp.newaxis, :] + step_costs
        child_depths = parent_depths + 1

        flat_size = action_size * batch_size
        flat_neighbours = xnp.reshape(neighbours, (flat_size,))
        flat_g = child_costs.reshape((flat_size,))
        flat_depth = jnp.tile(child_depths, (action_size,)).reshape((flat_size,))
        flat_valid = jnp.logical_and(jnp.tile(valid_mask, (action_size,)), jnp.isfinite(flat_g))

        # Actions bit
        flat_actions = jnp.tile(
            jnp.arange(action_size, dtype=jnp.int32)[:, None], (1, batch_size)
        ).reshape((flat_size,))

        # --- In-Batch Deduplication ---
        unique_mask = xnp.unique_mask(flat_neighbours, key=flat_g, filled=flat_valid)
        flat_valid = jnp.logical_and(flat_valid, unique_mask)
        # -----------------------------

        # 4. Step Decision
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
                variable_q_step_switcher,
            ),
            sr_solved,
        )

        return loop_state.replace(search_result=return_sr)

    def _expand_step(sr, states, gs, depths, actions, valid, q_params, q_fn):
        # Compact for Q evaluation
        sort_keys_pre = jnp.where(valid, 0, 1)
        perm = jnp.argsort(sort_keys_pre)

        states_sorted = states[perm]
        gs_sorted = gs[perm]
        depths_sorted = depths[perm]
        actions_sorted = actions[perm]
        valid_sorted = valid[perm]

        # Evaluate Q on compacted front
        hs_sorted = q_fn(q_params, states_sorted, valid_sorted).astype(KEY_DTYPE)
        fs_sorted = (cost_weight * gs_sorted + hs_sorted).astype(KEY_DTYPE)

        active_bound = sr.bound
        keep_mask_sorted = jnp.logical_and(valid_sorted, fs_sorted <= active_bound + 1e-6)
        prune_mask_sorted = jnp.logical_and(valid_sorted, fs_sorted > active_bound + 1e-6)

        pruned_fs = jnp.where(prune_mask_sorted, fs_sorted, jnp.inf)
        min_pruned = jnp.min(pruned_fs).astype(KEY_DTYPE)

        new_next_bound = jnp.minimum(sr.next_bound, min_pruned).astype(KEY_DTYPE)
        sr = sr.replace(next_bound=new_next_bound)

        return sr.push_batch(
            states_sorted, gs_sorted, depths_sorted, actions_sorted, keep_mask_sorted
        )

    # 4. Outer Loop
    def outer_cond(loop_state: IDLoopState):
        sr = loop_state.search_result
        return jnp.logical_and(~sr.solved, jnp.isfinite(sr.bound))

    def outer_body(loop_state: IDLoopState):
        # Run DFS
        loop_state = jax.lax.while_loop(inner_cond, inner_body, loop_state)

        sr = loop_state.search_result
        new_bound = sr.next_bound

        # Reset but keep frontier
        reset_sr = sr.replace(
            bound=new_bound,
            next_bound=jnp.array(jnp.inf, dtype=KEY_DTYPE),
            stack_ptr=jnp.array(0, dtype=jnp.int32),
        )

        # Restart from frontier
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

    if show_compile_time:
        print("initializing jit for ID-Q*")
        start_t = time.time()

    # Optional dry run can be added here if needed, similar to IDA*

    if show_compile_time:
        print(f"JIT compile time: {time.time() - start_t:.2f}s")

    return id_qstar_fn
