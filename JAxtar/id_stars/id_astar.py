import time

import jax
import jax.numpy as jnp
import xtructure.numpy as xnp
from puxle import Puzzle

from heuristic.heuristic_base import Heuristic
from JAxtar.annotate import KEY_DTYPE, MIN_BATCH_SIZE
from JAxtar.id_stars.search_base import IDFrontier, IDLoopState, IDSearchResult
from JAxtar.utils.batch_switcher import variable_batch_switcher_builder


def _id_astar_frontier_builder(
    puzzle: Puzzle,
    heuristic: Heuristic,
    batch_size: int = 1024,
):
    """
    Returns a function that generates an initial frontier starting from a single state.
    Uses Batched BFS with in-batch deduplication. (Inlined for ID-A*)
    """
    action_size = puzzle.action_size
    max_children = batch_size * action_size

    variable_heuristic = variable_batch_switcher_builder(
        heuristic.batched_distance,
        max_batch_size=max_children,
        min_batch_size=MIN_BATCH_SIZE,
        pad_value=jnp.inf,
    )

    def generate_frontier(
        solve_config: Puzzle.SolveConfig, start: Puzzle.State, **kwargs
    ) -> IDFrontier:
        if "h_params" in kwargs:
            h_params = kwargs["h_params"]
        else:
            h_params = heuristic.prepare_heuristic_parameters(solve_config, **kwargs)

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
                not_solved, jnp.logical_and(within_limit, jnp.logical_and(has_capacity, has_nodes))
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

            sort_keys_pre = jnp.where(flat_valid, 0, 1)
            perm_pre = jnp.argsort(sort_keys_pre)

            states_pre = flat_states[perm_pre]
            gs_pre = flat_g[perm_pre]
            depths_pre = flat_depth[perm_pre]
            valid_pre = flat_valid[perm_pre]

            hs_pre = variable_heuristic(h_params, states_pre, valid_pre).astype(KEY_DTYPE)
            fs_pre = gs_pre + hs_pre

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

    generate_frontier = _id_astar_frontier_builder(puzzle, heuristic, batch_size)

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

        states_sorted = xnp.take(states, perm, axis=0)
        gs_sorted = xnp.take(gs, perm, axis=0)
        depths_sorted = xnp.take(depths, perm, axis=0)
        actions_sorted = xnp.take(actions, perm, axis=0)
        valid_sorted = xnp.take(valid, perm, axis=0)

        hs_sorted = h_fn(h_params, states_sorted, valid_sorted).astype(KEY_DTYPE)
        fs_sorted = (cost_weight * gs_sorted + hs_sorted).astype(KEY_DTYPE)

        active_bound = sr.bound

        keep_mask_sorted = jnp.logical_and(valid_sorted, fs_sorted <= active_bound + 1e-6)

        # Optimization: Sort valid children by f-value descending (Worst -> Best)
        # We construct a key such that Valid items come first, and within Valid, sorted by F descending.
        f_key = jnp.where(keep_mask_sorted, -fs_sorted, jnp.inf)
        perm_f = jnp.argsort(f_key)

        states_ordered = xnp.take(states_sorted, perm_f, axis=0)
        gs_ordered = xnp.take(gs_sorted, perm_f, axis=0)
        depths_ordered = xnp.take(depths_sorted, perm_f, axis=0)
        actions_ordered = xnp.take(actions_sorted, perm_f, axis=0)

        n_push = jnp.sum(keep_mask_sorted)

        prune_mask_sorted = jnp.logical_and(valid_sorted, fs_sorted > active_bound + 1e-6)
        pruned_fs = jnp.where(prune_mask_sorted, fs_sorted, jnp.inf)
        min_pruned_f = jnp.min(pruned_fs).astype(KEY_DTYPE)
        new_next_bound = jnp.minimum(sr.next_bound, min_pruned_f).astype(KEY_DTYPE)

        sr_next = sr.replace(next_bound=new_next_bound)

        return sr_next.push_packed_batch(
            states_ordered,
            gs_ordered,
            depths_ordered,
            actions_ordered,
            n_push,
        )

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
            stack=sr.stack.replace(size=jnp.array(0, dtype=jnp.uint32)),
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
