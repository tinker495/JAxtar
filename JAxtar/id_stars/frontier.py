import jax
import jax.numpy as jnp
import xtructure.numpy as xnp
from puxle import Puzzle

from heuristic.heuristic_base import Heuristic
from JAxtar.annotate import KEY_DTYPE, MIN_BATCH_SIZE
from JAxtar.id_stars.search_base import IDFrontier
from JAxtar.utils.batch_switcher import variable_batch_switcher_builder


def frontier_builder(
    puzzle: Puzzle,
    heuristic: Heuristic,
    batch_size: int = 1024,
):
    """
    Returns a function that generates a frontier starting from a single state.
    Uses Batched BFS with in-batch deduplication.
    """
    action_size = puzzle.action_size
    max_children = batch_size * action_size

    # 1. State Switcher for heuristic evaluation
    # We use this to only evaluate active states in the batch.
    variable_heuristic = variable_batch_switcher_builder(
        heuristic.batched_distance,
        max_batch_size=max_children,
        min_batch_size=MIN_BATCH_SIZE,
        pad_value=jnp.inf,
    )

    def generate_frontier(
        solve_config: Puzzle.SolveConfig, start: Puzzle.State, **kwargs
    ) -> IDFrontier:
        # Determine h_params
        # If 'h_params' in kwargs, use it. Otherwise prepare.
        if "h_params" in kwargs:
            h_params = kwargs["h_params"]
        else:
            h_params = heuristic.prepare_heuristic_parameters(solve_config, **kwargs)

        # 1. Initialize padded arrays
        # We start with [1, ...] then expand to [batch_size, ...]
        start_reshaped = xnp.expand_dims(start, axis=0)  # [1, state_shape]

        # Check if root is solved
        root_solved = puzzle.batched_is_solved(solve_config, start_reshaped)[0]

        # Initial Frontier
        start_padded = jax.tree_util.tree_map(
            lambda x: jnp.pad(x, [(0, batch_size - 1)] + [(0, 0)] * (x.ndim - 1)), start_reshaped
        )

        costs_padded = jnp.full((batch_size,), jnp.inf, dtype=KEY_DTYPE)
        costs_padded = costs_padded.at[0].set(0.0)
        depths_padded = jnp.full((batch_size,), 0, dtype=jnp.int32)
        valid_padded = jnp.zeros((batch_size,), dtype=jnp.bool_)
        valid_padded = valid_padded.at[0].set(True)

        solution_state = start_reshaped  # if root_solved is False, this is just a placeholder
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

        # 2. While loop for BFS expansion
        # We continue until the frontier is full (num_valid >= batch_size) OR we exceed a step limit
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

            # 1. Expand all valid nodes in current frontier
            states = frontier.states
            gs = frontier.costs
            depth = frontier.depths
            valid = frontier.valid_mask

            neighbours, step_costs = puzzle.batched_get_neighbours(solve_config, states, valid)
            # neighbours: [action_size, batch_size, ...]
            # step_costs: [action_size, batch_size]

            child_g = (gs[jnp.newaxis, :] + step_costs).astype(KEY_DTYPE)
            child_depth = depth + 1

            flat_size = action_size * batch_size
            flat_states = xnp.reshape(neighbours, (flat_size,))
            flat_g = child_g.reshape((flat_size,))
            flat_depth = jnp.tile(child_depth, (action_size,)).reshape((flat_size,))

            # Validity
            flat_parent_valid = jnp.tile(valid, (action_size,))
            flat_valid = jnp.logical_and(flat_parent_valid, jnp.isfinite(flat_g))

            # 2. Check Solved
            is_solved_mask = puzzle.batched_is_solved(solve_config, flat_states)
            is_solved_mask = jnp.logical_and(is_solved_mask, flat_valid)
            any_solved = jnp.any(is_solved_mask)

            # Update solution info if found (pick first)
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

            # 3. Compaction before Heuristic Evaluation
            # We must put valid items at the front because variable_heuristic expects it.
            # Use argsort to move valid nodes to the front.
            sort_keys_pre = jnp.where(flat_valid, 0, 1)
            perm_pre = jnp.argsort(sort_keys_pre)

            states_pre = flat_states[perm_pre]
            gs_pre = flat_g[perm_pre]
            depths_pre = flat_depth[perm_pre]
            valid_pre = flat_valid[perm_pre]

            # 4. Heuristic Evaluation (on compacted nodes)
            hs_pre = variable_heuristic(h_params, states_pre, valid_pre).astype(KEY_DTYPE)
            fs_pre = gs_pre + hs_pre

            # 5. Deduplication (In-Batch)
            # We use xnp.unique_mask to remove duplicates in the current expansion.
            # Key = flat_g (or state hash if available, but g is safe for SlidePuzzle).
            unique_mask_pre = xnp.unique_mask(states_pre, key=gs_pre, filled=valid_pre)
            valid_after_dedup = jnp.logical_and(valid_pre, unique_mask_pre)

            # 6. Final Selection / Compaction
            # Best nodes (lowest F) should be at the front.
            # Partition: Valid nodes with finite F first, then Invalid.
            # Offset invalid nodes by a large amount.
            # Use nan_to_num to handle any potential inf/nan.
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
