from typing import Any

import jax
import jax.numpy as jnp
import xtructure.numpy as xnp
from puxle import Puzzle

from helpers.jax_compile import jit_with_warmup
from JAxtar.annotate import ACTION_DTYPE, KEY_DTYPE, MIN_BATCH_SIZE
from JAxtar.stars.search_base import (
    Current,
    LoopStateWithStates,
    Parant_with_Costs,
    Parent,
    SearchResult,
    finalize_search_result,
    insert_priority_queue_batches,
    loop_continue_if_not_solved,
    sort_and_pack_action_candidates,
)
from JAxtar.utils.batch_switcher import variable_batch_switcher_builder
from qfunction.q_base import QFunction


def _qstar_loop_builder(
    puzzle: Puzzle,
    q_fn: QFunction,
    batch_size: int = 1024,
    max_nodes: int = int(1e6),
    pop_ratio: float = jnp.inf,
    cost_weight: float = 1.0 - 1e-6,
    look_ahead_pruning: bool = True,
    pessimistic_update: bool = True,
):
    # The loop builder keeps init/cond/body reusable so callers can
    # access mid-loop data (params, queue/hash state) without rewriting
    # the search plumbing or retracing the whole loop.
    statecls = puzzle.State
    action_size = puzzle.action_size

    dist_sign = -1.0 if pessimistic_update else 1.0
    denom = max(1, puzzle.action_size // 2)
    min_pop = max(1, MIN_BATCH_SIZE // denom)

    # min_pop determines the minimum number of states to pop from the priority queue in each batch.
    # This value is set to optimize the efficiency of batched operations.
    # By ensuring that at least this many states are processed together,
    # we maximize parallelism and hardware utilization,
    # which is especially important for JAX and accelerator-based computation.
    # The formula (batch_size // (puzzle.action_size // 2)) is chosen to balance the number of expansions per batch,
    # so that each batch is filled as evenly as possible and computational resources are used efficiently.
    variable_q_batch_switcher = variable_batch_switcher_builder(
        q_fn.batched_q_value,
        max_batch_size=batch_size,
        min_batch_size=MIN_BATCH_SIZE,
        pad_value=jnp.inf,
    )

    def init_loop_state(solve_config: Puzzle.SolveConfig, start: Puzzle.State, **kwargs):
        search_result: SearchResult = SearchResult.build(
            statecls,
            batch_size,
            max_nodes,
            action_size,
            pop_ratio=pop_ratio,
            min_pop=min_pop,
            parant_with_costs=True,
        )
        q_parameters = q_fn.prepare_q_parameters(solve_config, **kwargs)

        (
            search_result.hashtable,
            _,
            hash_idx,
        ) = search_result.hashtable.insert(start)

        search_result.cost = search_result.cost.at[hash_idx.index].set(0)
        sr_batch_size = search_result.batch_size
        costs = jnp.zeros((sr_batch_size,), dtype=KEY_DTYPE)
        states = xnp.pad(start, (0, sr_batch_size - 1))
        hash_idxs = xnp.pad(hash_idx, (0, sr_batch_size - 1))
        filled = jnp.zeros(sr_batch_size, dtype=jnp.bool_).at[0].set(True)

        loop_state = LoopStateWithStates(
            search_result=search_result,
            solve_config=solve_config,
            params=q_parameters,
            current=Current(hashidx=hash_idxs, cost=costs),
            states=states,
            filled=filled,
        )
        return loop_state

    def loop_condition(loop_state: LoopStateWithStates):
        search_result = loop_state.search_result
        solve_config = loop_state.solve_config
        states = loop_state.states
        filled = loop_state.filled
        return loop_continue_if_not_solved(search_result, puzzle, solve_config, states, filled)

    def loop_body(loop_state: LoopStateWithStates):
        search_result = loop_state.search_result
        solve_config = loop_state.solve_config
        q_parameters = loop_state.params
        cost = loop_state.current.cost
        hash_idx = loop_state.current.hashidx
        filled = loop_state.filled
        states = loop_state.states

        action_size = search_result.action_size
        sr_batch_size = search_result.batch_size
        idx_tiles = xnp.tile(hash_idx, (action_size, 1))  # [action_size, batch_size, ...]
        action = jnp.tile(
            jnp.arange(action_size, dtype=ACTION_DTYPE)[:, jnp.newaxis],
            (1, sr_batch_size),
        )  # [n_neighbours, batch_size]
        costs = jnp.tile(cost[jnp.newaxis, :], (action_size, 1))  # [action_size, batch_size]
        filled_tiles = jnp.tile(
            filled[jnp.newaxis, :], (action_size, 1)
        )  # [action_size, batch_size]

        # Compute Q-values for parent states (not neighbors)
        # This gives us Q(s, a) for all actions from parent states
        q_vals = variable_q_batch_switcher(q_parameters, states, filled)
        q_vals = q_vals.transpose().astype(KEY_DTYPE)  # [action_size, batch_size]

        neighbour_keys = (cost_weight * costs + q_vals).astype(KEY_DTYPE)
        neighbour_keys = jnp.where(filled_tiles, neighbour_keys, jnp.inf)

        flattened_filled_tiles = filled_tiles.flatten()
        flattened_keys = neighbour_keys.flatten()
        dists = q_vals.flatten()

        if look_ahead_pruning:
            # NOTE: Q* in its canonical form only evaluates parent states and relies on the
            # priority queue ordering to discover optimal frontiers. This look-ahead step
            # peeks at neighbour states before inserting them into the queue, effectively
            # performing an inexpensive expansion filter. That means this path is no longer
            # a textbook Q* implementation, but it becomes extremely cost-effective whenever
            # puzzle dynamics (the environment step) are far cheaper than pushing extra items
            # through the batched priority queue / hash-table machinery. When the simulator
            # dominates runtime, disable this block to remain faithful to vanilla Q*. The gain
            # is even larger in highly reversible environments (common in puzzles) where many
            # neighbour evaluations collapse to repeated configurations, so aggressive pruning
            # saves substantial queue churn.
            # Look-a-head pruning can be disabled to explore more neighbours.
            # Even without pruning, we still sort keys to minimize data-structure I/O.
            neighbour_look_a_head, ncosts = puzzle.batched_get_neighbours(
                solve_config, states, filled
            )  # [action_size, batch_size]
            look_a_head_costs = costs + ncosts  # [action_size, batch_size]
            flattened_neighbour_look_head = neighbour_look_a_head.flatten()
            flattened_look_a_head_costs = look_a_head_costs.flatten()

            # Prioritize min cost, then check pessimistic/optimistic Q
            # If pessimistic: we want larger dist (score = cost - eps * dist)
            # If optimistic: we want smaller dist (score = cost + eps * dist)
            distinct_score = flattened_look_a_head_costs + dist_sign * 1e-5 * dists

            unique_mask = xnp.unique_mask(
                flattened_neighbour_look_head,
                distinct_score,
                flattened_filled_tiles,
            )
            current_hash_idxs, found = search_result.hashtable.lookup_parallel(
                flattened_neighbour_look_head, unique_mask
            )
            old_costs = search_result.get_cost(current_hash_idxs)
            old_dists = search_result.get_dist(current_hash_idxs)

            # Update Q-values: use max (pessimistic) or min (optimistic) of new and existing
            if pessimistic_update:
                # `old_dists` is stored for the *state* after pop.
                # In deferred Q*, pop stores: dist(state) = Q(parent, action) - step_cost.
                # To compare/update in Q-space we reconstruct:
                #   Q_old(parent->child) = old_dist(child) + step_cost(parent->child)
                step_cost = ncosts.flatten().astype(KEY_DTYPE)
                q_old = old_dists.astype(KEY_DTYPE) + step_cost
                q_old_for_max = jnp.where(found, q_old, -jnp.inf)
                dists = jnp.maximum(dists, q_old_for_max)
            else:
                step_cost = ncosts.flatten().astype(KEY_DTYPE)
                q_old = old_dists.astype(KEY_DTYPE) + step_cost
                q_old_for_min = jnp.where(found, q_old, jnp.inf)
                dists = jnp.minimum(dists, q_old_for_min)

            # Only consider nodes that are either:
            # 1. Not found in the hash table (new nodes), or
            # 2. Found but have better cost than existing
            # Note: If unique_mask is False, found is also False, so we can optimize by checking unique_mask first
            better_cost_mask = jnp.less(flattened_look_a_head_costs, old_costs)
            optimal_mask = unique_mask & (jnp.logical_or(~found, better_cost_mask))
        else:
            optimal_mask = flattened_filled_tiles

        flattened_vals = Parant_with_Costs(
            parent=Parent(hashidx=idx_tiles.flatten(), action=action.flatten()),
            cost=costs.flatten(),
            dist=dists,
        )

        neighbour_keys, vals, optimal_mask = sort_and_pack_action_candidates(
            flattened_keys,
            flattened_vals,
            optimal_mask,
            action_size,
            sr_batch_size,
        )

        search_result = insert_priority_queue_batches(
            search_result,
            neighbour_keys,
            vals,
            optimal_mask,
        )
        search_result, min_val, next_states, filled = search_result.pop_full_with_actions(
            puzzle=puzzle, solve_config=solve_config
        )

        return LoopStateWithStates(
            search_result=search_result,
            solve_config=solve_config,
            params=q_parameters,
            current=min_val,
            states=next_states,
            filled=filled,
        )

    return init_loop_state, loop_condition, loop_body


def qstar_builder(
    puzzle: Puzzle,
    q_fn: QFunction,
    batch_size: int = 1024,
    max_nodes: int = int(1e6),
    pop_ratio: float = jnp.inf,
    cost_weight: float = 1.0 - 1e-6,
    show_compile_time: bool = False,
    look_ahead_pruning: bool = True,
    pessimistic_update: bool = True,
    warmup_inputs: tuple[Puzzle.SolveConfig, Puzzle.State] | None = None,
):
    """
    Builds and returns a JAX-accelerated Q* search function.

    Args:
        puzzle: Puzzle instance that defines the problem space and operations.
        q_fn: QFunction instance that provides state-action value estimation.
        batch_size: Number of states to process in parallel (default: 1024).
        max_nodes: Maximum number of nodes to explore before terminating (default: 1e6).
        pop_ratio: Ratio of states to pop from the priority queue.
        cost_weight: Weight applied to the path cost in the Q* algorithm (default: 1.0-1e-6).
                    Values closer to 1.0 make the search more greedy/depth-first.
        show_compile_time: If True, displays the time taken to compile the search function (default: False).
        look_ahead_pruning: If True, enables neighbour look-ahead pruning for duplicate and cost-based filtering.
        pessimistic_update: If True, maintains the maximum (pessimistic) Q-value when a duplicate state is found.
                           If False, uses the minimum (optimistic) Q-value. Default is True.

    Returns:
        A function that performs Q* search given a start state and solve configuration.
    """
    init_loop_state, loop_condition, loop_body = _qstar_loop_builder(
        puzzle,
        q_fn,
        batch_size,
        max_nodes,
        pop_ratio,
        cost_weight,
        look_ahead_pruning,
        pessimistic_update,
    )

    def qstar(
        solve_config: Puzzle.SolveConfig,
        start: Puzzle.State,
        **kwargs: Any,
    ) -> SearchResult:
        """
        qstar is the implementation of the Q* algorithm.
        """
        loop_state = init_loop_state(solve_config, start, **kwargs)
        loop_state = jax.lax.while_loop(loop_condition, loop_body, loop_state)

        search_result = loop_state.search_result
        current = loop_state.current
        states = loop_state.states
        filled = loop_state.filled

        solved_mask = jnp.logical_and(puzzle.batched_is_solved(solve_config, states), filled)
        return finalize_search_result(search_result, current, solved_mask)

    return jit_with_warmup(
        qstar,
        puzzle=puzzle,
        show_compile_time=show_compile_time,
        warmup_inputs=warmup_inputs,
    )
