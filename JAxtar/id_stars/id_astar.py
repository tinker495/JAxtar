import time

import jax
import jax.numpy as jnp
import xtructure.numpy as xnp
from puxle import Puzzle

from heuristic.heuristic_base import Heuristic
from JAxtar.annotate import ACTION_DTYPE, KEY_DTYPE, MIN_BATCH_SIZE
from JAxtar.id_stars.search_base import (
    ACTION_PAD,
    IDFrontier,
    IDLoopState,
    IDSearchResult,
    compact_by_valid,
)
from JAxtar.id_stars.utils import _apply_non_backtracking, build_id_node_batch
from JAxtar.utils.batch_switcher import variable_batch_switcher_builder


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

    if non_backtracking_steps < 0:
        raise ValueError("non_backtracking_steps must be non-negative")
    non_backtracking_steps = int(non_backtracking_steps)

    IDNodeBatch = build_id_node_batch(statecls, non_backtracking_steps, max_path_len)

    variable_heuristic = variable_batch_switcher_builder(
        heuristic.batched_distance,
        max_batch_size=batch_size,
        min_batch_size=MIN_BATCH_SIZE,
        pad_value=jnp.inf,
    )
    flat_indices = jnp.arange(flat_size, dtype=jnp.int32)
    trail_indices = jnp.arange(non_backtracking_steps, dtype=jnp.int32)

    def _chunked_heuristic_eval(
        h_params: Puzzle.SolveConfig,
        flat_states: Puzzle.State,
        flat_valid: jnp.ndarray,
    ) -> jnp.ndarray:
        sort_key = jnp.logical_not(flat_valid).astype(jnp.int32)
        _, sorted_idx = jax.lax.sort_key_val(sort_key, flat_indices, dimension=0, is_stable=True)
        sorted_states = xnp.take(flat_states, sorted_idx, axis=0)
        sorted_mask = flat_valid[sorted_idx]

        chunk_states = xnp.reshape(sorted_states, (action_size, batch_size))
        chunk_mask = sorted_mask.reshape((action_size, batch_size))

        def _compute(_, inputs):
            states_slice, mask_slice = inputs

            def _calc(_):
                vals = variable_heuristic(h_params, states_slice, mask_slice).astype(KEY_DTYPE)
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
        trail_padded = statecls.default((batch_size, non_backtracking_steps))

        costs_padded = jnp.full((batch_size,), jnp.inf, dtype=KEY_DTYPE)
        costs_padded = costs_padded.at[0].set(0.0)
        depths_padded = jnp.full((batch_size,), 0, dtype=jnp.int32)
        valid_padded = jnp.zeros((batch_size,), dtype=jnp.bool_)
        valid_padded = valid_padded.at[0].set(True)

        action_history_padded = jnp.full((batch_size, max_path_len), ACTION_PAD, dtype=ACTION_DTYPE)
        action_ids = jnp.arange(action_size, dtype=jnp.int32)

        solution_state = start_reshaped
        solution_cost = jax.lax.cond(
            root_solved,
            lambda _: jnp.array(0, dtype=KEY_DTYPE),
            lambda _: jnp.array(jnp.inf, dtype=KEY_DTYPE),
            None,
        )
        solution_actions = jnp.full((max_path_len,), ACTION_PAD, dtype=ACTION_DTYPE)

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
                not_solved, jnp.logical_and(within_limit, jnp.logical_and(has_capacity, has_nodes))
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
                action_history[None, :, :], (action_size, batch_size, max_path_len)
            )
            flat_action_history = flat_action_history.reshape((flat_size, max_path_len))
            flat_actions = jnp.repeat(action_ids, batch_size)

            # Update history: insert action at current depth
            flat_depth_int = depth.astype(jnp.int32)
            flat_depth_tiled = jnp.broadcast_to(flat_depth_int[None, :], (action_size, batch_size))
            flat_depth_flat = flat_depth_tiled.reshape((flat_size,))

            # Clamp depth to prevent out-of-bounds writes (nodes exceeding max_path_len are pruned later)
            safe_depth = jnp.minimum(flat_depth_flat, max_path_len - 1)
            flat_action_history = flat_action_history.at[jnp.arange(flat_size), safe_depth].set(
                flat_actions.astype(ACTION_DTYPE)
            )

            flat_states = xnp.reshape(neighbours, (flat_size,))
            flat_g = child_g.reshape((flat_size,))
            flat_depth = jnp.broadcast_to(child_depth, (action_size, batch_size)).reshape(
                (flat_size,)
            )

            flat_parent_valid = jnp.broadcast_to(valid, (action_size, batch_size)).reshape(
                (flat_size,)
            )
            flat_valid = jnp.logical_and(flat_parent_valid, jnp.isfinite(flat_g))

            flat_action_history = jnp.where(
                flat_valid[:, None],
                flat_action_history,
                jnp.full_like(flat_action_history, ACTION_PAD),
            )

            is_solved_mask = puzzle.batched_is_solved(solve_config, flat_states)
            is_solved_mask = jnp.logical_and(is_solved_mask, flat_valid)
            any_solved = jnp.any(is_solved_mask)

            first_idx = jnp.argmax(is_solved_mask)
            found_sol_state = xnp.take(flat_states, first_idx[jnp.newaxis], axis=0)
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

            unique_mask = xnp.unique_mask(flat_states, key=flat_g, filled=flat_valid)
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

            packed_batch, packed_valid, valid_count, packed_idx = compact_by_valid(
                flat_batch, flat_valid
            )
            packed_f = jnp.where(packed_valid, f_safe[packed_idx], jnp.inf)

            neg_f = -packed_f
            top_vals, top_indices = jax.lax.top_k(neg_f, batch_size)
            selected_f = (-top_vals).astype(KEY_DTYPE)
            selected_valid = jnp.isfinite(selected_f)

            selected = xnp.take(packed_batch, top_indices, axis=0)
            selected_valid = jnp.logical_and(selected_valid, packed_valid[top_indices])

            new_frontier = IDFrontier(
                states=selected.state,
                costs=selected.cost,
                depths=selected.depth,
                valid_mask=selected_valid,
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
    if non_backtracking_steps < 0:
        raise ValueError("non_backtracking_steps must be non-negative")
    non_backtracking_steps = int(non_backtracking_steps)
    trail_indices = jnp.arange(non_backtracking_steps, dtype=jnp.int32)
    action_ids = jnp.arange(action_size, dtype=jnp.int32)
    flat_actions = jnp.broadcast_to(action_ids[:, None], (action_size, batch_size)).reshape(
        (flat_size,)
    )
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

    # Heuristic for the full frontier batch (size = batch_size)
    frontier_heuristic_fn = variable_batch_switcher_builder(
        heuristic.batched_distance,
        max_batch_size=batch_size,
        min_batch_size=MIN_BATCH_SIZE,
        pad_value=jnp.inf,
    )

    flat_indices = jnp.arange(flat_size, dtype=jnp.int32)

    def _chunked_heuristic_eval(
        h_params: Puzzle.SolveConfig,
        flat_states: Puzzle.State,
        flat_valid: jnp.ndarray,
    ) -> jnp.ndarray:
        sort_key = jnp.logical_not(flat_valid).astype(jnp.int32)
        _, sorted_idx = jax.lax.sort_key_val(sort_key, flat_indices, dimension=0, is_stable=True)
        sorted_states = xnp.take(flat_states, sorted_idx, axis=0)
        sorted_mask = flat_valid[sorted_idx]

        chunk_states = xnp.reshape(sorted_states, (action_size, batch_size))
        chunk_mask = sorted_mask.reshape((action_size, batch_size))

        def _compute(_, inputs):
            states_slice, mask_slice = inputs

            def _calc(_):
                vals = variable_heuristic_batch_switcher(h_params, states_slice, mask_slice).astype(
                    KEY_DTYPE
                )
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
        search_result = search_result.replace(frontier_action_history=frontier.action_history)
        # Calculate F values for the frontier to determine initial bound
        frontier_h = frontier_heuristic_fn(
            heuristic_parameters, frontier.states, frontier.valid_mask
        ).astype(KEY_DTYPE)
        frontier_f = (cost_weight * frontier.costs + frontier_h).astype(KEY_DTYPE)
        frontier = frontier.replace(f_scores=frontier_f)

        # Initial Bound = min(f) of valid frontier nodes
        # Filter invalid/inf
        valid_fs = jnp.where(frontier.valid_mask, frontier.f_scores, jnp.inf)
        start_bound = jnp.min(valid_fs).astype(KEY_DTYPE)

        # Initialize IDSearchResult with this bound
        # Also carry over solved status if frontier found solution
        search_result = search_result.replace(
            bound=start_bound,
            next_bound=jnp.array(jnp.inf, dtype=KEY_DTYPE),
            solved=frontier.solved,
            solution_state=frontier.solution_state,
            solution_cost=frontier.solution_cost,
            solution_actions_arr=frontier.solution_actions_arr,
            solved_idx=jnp.where(frontier.solved, -1, -1),
        )

        # ----------------------------------------------------------------------
        # 2. Push Initial Frontier (Subset <= Bound)
        # ----------------------------------------------------------------------
        search_result = _push_frontier_to_stack(search_result, frontier, start_bound)

        return IDLoopState(
            search_result=search_result,
            solve_config=solve_config,
            params=heuristic_parameters,
            frontier=frontier,
        )

    # Helper to push allowed frontier nodes to stack and update next_bound
    def _push_frontier_to_stack(sr, frontier, bound):
        fs = frontier.f_scores
        keep_mask = jnp.logical_and(frontier.valid_mask, fs <= bound + 1e-6)

        # Determine what to start next_bound with
        prune_mask = jnp.logical_and(frontier.valid_mask, fs > bound + 1e-6)
        pruned_fs = jnp.where(prune_mask, fs, jnp.inf)
        min_pruned = jnp.min(pruned_fs).astype(KEY_DTYPE)

        new_next_bound = jnp.minimum(sr.next_bound, min_pruned).astype(KEY_DTYPE)
        sr = sr.replace(next_bound=new_next_bound)

        parent_indices = jnp.full((batch_size,), -1, dtype=jnp.int32)
        root_indices = jnp.where(frontier.valid_mask, jnp.arange(batch_size), -1)
        return sr.push_batch(
            frontier.states,
            frontier.costs,
            frontier.depths,
            frontier_actions,
            parent_indices,
            root_indices,
            frontier.trail,
            frontier.action_history,
            keep_mask,
        )

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
        (
            sr,
            parents,
            parent_costs,
            parent_depths,
            parent_trails,
            parent_action_histories,
            valid_mask,
            parent_trace_indices,
            parent_root_indices,
        ) = sr.get_top_batch(batch_size)

        # 2. Check Solved
        is_solved_mask = puzzle.batched_is_solved(solve_config, parents)
        is_solved_mask = jnp.logical_and(is_solved_mask, valid_mask)
        any_solved = jnp.any(is_solved_mask)

        # If solved, store the solution state.
        first_solved_idx = jnp.argmax(is_solved_mask)
        solved_st_batch = xnp.take(parents, first_solved_idx[jnp.newaxis], axis=0)
        solved_cost = parent_costs[first_solved_idx]
        solved_actions = parent_action_histories[first_solved_idx]
        solved_trace_idx = parent_trace_indices[first_solved_idx]

        new_solution_state = jax.lax.cond(
            any_solved, lambda _: solved_st_batch, lambda _: sr.solution_state, None
        )
        new_solution_cost = jax.lax.cond(
            any_solved, lambda _: solved_cost.astype(KEY_DTYPE), lambda _: sr.solution_cost, None
        )
        new_solution_actions = jax.lax.cond(
            any_solved, lambda _: solved_actions, lambda _: sr.solution_actions_arr, None
        )

        sr_solved = sr.replace(
            solved=jnp.logical_or(sr.solved, any_solved),
            solved_idx=jnp.where(any_solved, solved_trace_idx, -1),
            solution_state=new_solution_state,
            solution_cost=new_solution_cost,
            solution_actions_arr=new_solution_actions,
        )

        # 3. Expand
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
            parent_action_histories[None, :, :], (action_size, batch_size, max_path_len)
        )
        flat_action_history = flat_action_history.reshape((flat_size, max_path_len))

        flat_depth_int = parent_depths.astype(jnp.int32)
        flat_depth_tiled = jnp.broadcast_to(flat_depth_int[None, :], (action_size, batch_size))
        flat_depth_flat = flat_depth_tiled.reshape((flat_size,))

        # Clamp depth to prevent out-of-bounds writes (nodes exceeding max_path_len are pruned later)
        safe_depth = jnp.minimum(flat_depth_flat, max_path_len - 1)
        flat_action_history = flat_action_history.at[jnp.arange(flat_size), safe_depth].set(
            flat_actions.astype(ACTION_DTYPE)
        )

        flat_neighbours = xnp.reshape(neighbours, (flat_size,))
        flat_g = child_costs.reshape((flat_size,))
        flat_depth = jnp.broadcast_to(child_depths, (action_size, batch_size)).reshape((flat_size,))

        flat_valid_parent = jnp.broadcast_to(valid_mask, (action_size, batch_size)).reshape(
            (flat_size,)
        )
        flat_valid = jnp.logical_and(flat_valid_parent, jnp.isfinite(flat_g))
        flat_valid = jnp.logical_and(flat_valid, flat_depth <= max_path_len)

        flat_parent_indices = jnp.tile(parent_trace_indices, action_size)
        flat_parent_indices = jnp.where(flat_valid, flat_parent_indices, -1)
        flat_root_indices = jnp.tile(parent_root_indices, action_size)
        flat_root_indices = jnp.where(flat_valid, flat_root_indices, -1)

        flat_action_history = jnp.where(
            flat_valid[:, None], flat_action_history, jnp.full_like(flat_action_history, ACTION_PAD)
        )

        # --- Optimization: In-Batch Deduplication ---
        # We perform uniqueness check on the generated flat batch.
        unique_mask = xnp.unique_mask(
            flat_neighbours,
            key=flat_g,  # Key: cost (keep lowest cost if duplicates)
            filled=flat_valid,
        )
        flat_valid = jnp.logical_and(flat_valid, unique_mask)

        flat_action_history = jnp.where(
            flat_valid[:, None], flat_action_history, jnp.full_like(flat_action_history, ACTION_PAD)
        )

        # --- Optimization: Global Deduplication (Transposition Table) ---
        (
            new_hashtable,
            is_new_mask,
            is_optimal_mask,
            hash_idx,
        ) = sr.hashtable.parallel_insert(flat_neighbours, flat_valid, flat_g)

        # Update hashtable in search result
        sr = sr.replace(hashtable=new_hashtable)

        # Update best known costs (g-value) in persistent storage
        new_ht_cost = xnp.update_on_condition(sr.ht_cost, hash_idx.index, is_optimal_mask, flat_g)
        sr = sr.replace(ht_cost=new_ht_cost)

        # Prune states that are not optimal (i.e., we found a cheaper or equal path before)
        flat_valid = jnp.logical_and(flat_valid, is_optimal_mask)

        flat_action_history = jnp.where(
            flat_valid[:, None], flat_action_history, jnp.full_like(flat_action_history, ACTION_PAD)
        )

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

        flat_action_history = jnp.where(
            flat_valid[:, None], flat_action_history, jnp.full_like(flat_action_history, ACTION_PAD)
        )
        # ----------------------------------------------------

        # Heuristic Caching & Evaluation
        # We only compute heuristic for NEW optimal states.
        # For existing optimal states, we fetch the cached heuristic.

        # Fetch cached heuristics
        old_h = sr.ht_dist[hash_idx.index]

        # Identify which heuristics need computation
        needs_h_mask = jnp.logical_and(flat_valid, is_new_mask)

        computed_h = _chunked_heuristic_eval(params, flat_neighbours, needs_h_mask)
        computed_h = jnp.maximum(0.0, computed_h)

        # Combine: use computed if new, else cached
        flat_h = jnp.where(is_new_mask, computed_h, old_h)

        # Cache new heuristics
        new_ht_dist = xnp.update_on_condition(sr.ht_dist, hash_idx.index, needs_h_mask, computed_h)
        sr = sr.replace(ht_dist=new_ht_dist)

        flat_f = (cost_weight * flat_g + flat_h).astype(KEY_DTYPE)

        # 4. Expansion
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
        active_bound = sr.bound
        keep_mask = jnp.logical_and(valid, fs <= active_bound + 1e-6)

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

        packed_batch, packed_valid, _, packed_idx = compact_by_valid(flat_batch, keep_mask)
        packed_fs = jnp.where(packed_valid, fs[packed_idx], jnp.inf)

        # Sort kept children by f-value descending (Worst -> Best) for LIFO stack order.
        f_key = jnp.where(packed_valid, -packed_fs, jnp.inf)
        perm_f = jnp.argsort(f_key)
        ordered = xnp.take(packed_batch, perm_f, axis=0)

        n_push = jnp.sum(packed_valid)

        prune_mask = jnp.logical_and(valid, fs > active_bound + 1e-6)
        pruned_fs = jnp.where(prune_mask, fs, jnp.inf)
        min_pruned_f = jnp.min(pruned_fs).astype(KEY_DTYPE)
        new_next_bound = jnp.minimum(sr.next_bound, min_pruned_f).astype(KEY_DTYPE)

        sr_next = sr.replace(next_bound=new_next_bound)

        return sr_next.push_packed_batch(
            ordered.state,
            ordered.cost,
            ordered.depth,
            ordered.action,
            ordered.parent_index,
            ordered.root_index,
            ordered.trail,
            ordered.action_history,
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
        ).reset_tables(statecls)

        # Push Frontier again with new bound
        reset_sr = _push_frontier_to_stack(reset_sr, loop_state.frontier, new_bound)

        return loop_state.replace(search_result=reset_sr)

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
