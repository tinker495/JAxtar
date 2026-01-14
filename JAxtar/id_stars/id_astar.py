import time

import jax
import jax.numpy as jnp
import xtructure.numpy as xnp
from puxle import Puzzle

from heuristic.heuristic_base import Heuristic
from JAxtar.annotate import KEY_DTYPE, MIN_BATCH_SIZE
from JAxtar.id_stars.frontier import frontier_builder
from JAxtar.id_stars.search_base import IDLoopState, IDSearchResult
from JAxtar.utils.batch_switcher import variable_batch_switcher_builder


def _id_astar_loop_builder(
    puzzle: Puzzle,
    heuristic: Heuristic,
    batch_size: int = 1024,
    max_nodes: int = int(1e6),
    cost_weight: float = 1.0,
):
    statecls = puzzle.State
    action_size = puzzle.action_size

    variable_heuristic_batch_switcher = variable_batch_switcher_builder(
        heuristic.batched_distance,
        max_batch_size=batch_size * action_size,
        min_batch_size=MIN_BATCH_SIZE,
        pad_value=jnp.inf,
    )

    generate_frontier = frontier_builder(puzzle, heuristic, batch_size)

    # Heuristic for the full frontier batch (size = batch_size)
    frontier_heuristic_fn = variable_batch_switcher_builder(
        heuristic.batched_distance,
        max_batch_size=batch_size,
        min_batch_size=MIN_BATCH_SIZE,
        pad_value=jnp.inf,
    )

    def init_loop_state(solve_config: Puzzle.SolveConfig, start: Puzzle.State, **kwargs):
        # Initialize Result
        search_result = IDSearchResult.build(
            statecls,
            capacity=max_nodes,
            action_size=action_size,
        )

        heuristic_parameters = heuristic.prepare_heuristic_parameters(solve_config, **kwargs)

        # ----------------------------------------------------------------------
        # 1. Generate Frontier
        # ----------------------------------------------------------------------
        frontier = generate_frontier(solve_config, start, **kwargs)

        # Calculate F values for the frontier to determine initial bound
        frontier_h = frontier_heuristic_fn(
            heuristic_parameters, frontier.states, frontier.valid_mask
        ).astype(KEY_DTYPE)
        frontier_f = (cost_weight * frontier.costs + frontier_h).astype(KEY_DTYPE)

        # Initial Bound = min(f) of valid frontier nodes
        # Filter invalid/inf
        valid_fs = jnp.where(frontier.valid_mask, frontier_f, jnp.inf)
        start_bound = jnp.min(valid_fs).astype(KEY_DTYPE)

        # Initialize IDSearchResult with this bound
        # Also carry over solved status if frontier found solution
        search_result = search_result.replace(
            bound=start_bound,
            next_bound=jnp.array(jnp.inf, dtype=KEY_DTYPE),
            solved=frontier.solved,
            solution_state=frontier.solution_state,
            solution_cost=frontier.solution_cost,
            solved_idx=jnp.where(frontier.solved, 0, -1),  # Dummy index
        )

        # ----------------------------------------------------------------------
        # 2. Push Initial Frontier (Subset <= Bound)
        # ----------------------------------------------------------------------
        def _push_frontier(sr, bound):
            # Re-calculate F (or could store it, but cheap to re-calc for batch_size)
            # We access frontier from closure or arg?
            # Passed as arg usually better for explicit dependency, but here we can close over 'frontier'
            # if we are inside init. But outer_body needs it too.
            # So we define a helper that takes 'frontier'.
            pass

        # We prefer a shared helper. Let's define it inside the builder scope or per-call.
        # Since it depends on 'frontier' which is dynamic data (from loop_state),
        # we define it to take frontier as input.

        search_result = _push_frontier_to_stack(
            search_result,
            frontier,
            frontier_heuristic_fn,
            heuristic_parameters,
            start_bound,
            batch_size,
            action_size,
        )

        return IDLoopState(
            search_result=search_result,
            solve_config=solve_config,
            params=heuristic_parameters,
            frontier=frontier,
        )

    # Helper to push allowed frontier nodes to stack and update next_bound
    def _push_frontier_to_stack(sr, frontier, h_fn, h_params, bound, batch_size, action_size):
        # Calc F
        hs = h_fn(h_params, frontier.states, frontier.valid_mask).astype(KEY_DTYPE)
        fs = (cost_weight * frontier.costs + hs).astype(KEY_DTYPE)

        # Determine what to push
        keep_mask = jnp.logical_and(frontier.valid_mask, fs <= bound + 1e-6)

        # Determine what to start next_bound with
        prune_mask = jnp.logical_and(frontier.valid_mask, fs > bound + 1e-6)
        pruned_fs = jnp.where(prune_mask, fs, jnp.inf)
        min_pruned = jnp.min(pruned_fs).astype(KEY_DTYPE)

        new_next_bound = jnp.minimum(sr.next_bound, min_pruned).astype(KEY_DTYPE)
        sr = sr.replace(next_bound=new_next_bound)

        # We need to construct 'actions' array for the frontier (default -1)
        actions = jnp.full((batch_size,), -1, dtype=jnp.int32)

        return sr.push_batch(frontier.states, frontier.costs, frontier.depths, actions, keep_mask)

    # -----------------------------------------------------------------------
    # INNER LOOP: Standard Batched DFS with Pruning by Bound
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

        # 1. Pop Batch
        sr, parents, parent_costs, parent_depths, valid_mask, _ = sr.get_top_batch(batch_size)

        # 2. Check Solved
        is_solved_mask = puzzle.batched_is_solved(solve_config, parents)
        is_solved_mask = jnp.logical_and(is_solved_mask, valid_mask)
        any_solved = jnp.any(is_solved_mask)

        # If solved, store the solution state.
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

        # 3. Expand
        neighbours, step_costs = puzzle.batched_get_neighbours(solve_config, parents, valid_mask)

        child_costs = parent_costs[jnp.newaxis, :] + step_costs
        child_depths = parent_depths + 1

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

        # --- Optimization: In-Batch Deduplication ---
        # We perform uniqueness check on the generated flat batch.
        # This prevents flooding the stack with redundant states from the same parent batch.
        unique_mask = xnp.unique_mask(
            flat_neighbours,
            key=flat_g,  # Key: cost (keep lowest cost if duplicates)
            filled=flat_valid,
        )
        flat_valid = jnp.logical_and(flat_valid, unique_mask)
        # ----------------------------------------------------

        # 4. Heuristic & Pruning
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
                variable_heuristic_batch_switcher,
            ),
            sr_solved,
        )

        return loop_state.replace(search_result=return_sr)

    def _expand_step(sr, states, gs, depths, actions, valid, h_params, h_fn):
        # Compact valid items to the front
        sort_keys = jnp.where(valid, 0, 1)
        perm = jnp.argsort(sort_keys)

        states_sorted = states[perm]
        gs_sorted = gs[perm]
        depths_sorted = depths[perm]
        actions_sorted = actions[perm]
        valid_sorted = valid[perm]

        hs_sorted = h_fn(h_params, states_sorted, valid_sorted).astype(KEY_DTYPE)
        fs_sorted = (cost_weight * gs_sorted + hs_sorted).astype(KEY_DTYPE)

        active_bound = sr.bound

        keep_mask_sorted = jnp.logical_and(valid_sorted, fs_sorted <= active_bound + 1e-6)
        prune_mask_sorted = jnp.logical_and(valid_sorted, fs_sorted > active_bound + 1e-6)

        pruned_fs = jnp.where(prune_mask_sorted, fs_sorted, jnp.array(jnp.inf, dtype=KEY_DTYPE))
        min_pruned_f = jnp.min(pruned_fs).astype(KEY_DTYPE)
        new_next_bound = jnp.minimum(sr.next_bound, min_pruned_f).astype(KEY_DTYPE)

        sr_next = sr.replace(next_bound=new_next_bound)

        sr_final = sr_next.push_batch(
            states_sorted, gs_sorted, depths_sorted, actions_sorted, keep_mask_sorted
        )
        return sr_final

    # -----------------------------------------------------------------------
    # OUTER LOOP: Iterative Deepening
    # -----------------------------------------------------------------------
    def outer_cond(loop_state: IDLoopState):
        sr = loop_state.search_result
        return jnp.logical_and(~sr.solved, jnp.isfinite(sr.bound))

    def outer_body(loop_state: IDLoopState):
        # 1. Run Inner Loop (DFS)
        loop_state = jax.lax.while_loop(inner_cond, inner_body, loop_state)

        sr = loop_state.search_result

        # 2. Update Bound & Reset
        new_bound = sr.next_bound

        reset_sr = sr.replace(
            bound=new_bound,
            next_bound=jnp.array(jnp.inf, dtype=KEY_DTYPE),
            stack_ptr=jnp.array(0, dtype=jnp.int32),
        )

        # Push Frontier again with new bound
        reset_sr = _push_frontier_to_stack(
            reset_sr,
            loop_state.frontier,
            frontier_heuristic_fn,
            loop_state.params,
            new_bound,
            batch_size,
            action_size,
        )

        return loop_state.replace(search_result=reset_sr)

    return init_loop_state, outer_cond, outer_body


def id_astar_builder(
    puzzle: Puzzle,
    heuristic: Heuristic,
    batch_size: int = 1024,
    max_nodes: int = int(1e6),
    cost_weight: float = 1.0,
    pop_ratio: float = 1.0,
    show_compile_time: bool = False,
):
    init_loop, cond, body = _id_astar_loop_builder(
        puzzle, heuristic, batch_size, max_nodes, cost_weight
    )

    def id_astar(solve_config: Puzzle.SolveConfig, start: Puzzle.State, **kwargs):
        loop_state = init_loop(solve_config, start, **kwargs)
        loop_state = jax.lax.while_loop(cond, body, loop_state)
        return loop_state.search_result

    id_astar_fn = jax.jit(id_astar)

    empty_solve_config = puzzle.SolveConfig.default()
    empty_states = puzzle.State.default()

    if show_compile_time:
        print("initializing jit")
        start_time = time.time()

    id_astar_fn(empty_solve_config, empty_states)

    if show_compile_time:
        end_time = time.time()
        print(f"Compile Time: {end_time - start_time:6.2f} seconds")
        print("JIT compiled\n\n")

    return id_astar_fn
