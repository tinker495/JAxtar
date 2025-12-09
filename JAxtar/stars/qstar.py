import time
from typing import Any

import jax
import jax.numpy as jnp
import xtructure.numpy as xnp
from puxle import Puzzle

from JAxtar.annotate import ACTION_DTYPE, KEY_DTYPE, MIN_BATCH_SIZE
from JAxtar.stars.search_base import (
    Current,
    LoopState,
    Parant_with_Costs,
    Parent,
    SearchResult,
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
        lambda solve_config, current: q_fn.batched_q_value(solve_config, current),
        max_batch_size=batch_size,
        min_batch_size=MIN_BATCH_SIZE,
        pad_value=jnp.inf,
    )

    def init_loop_state(solve_config: Puzzle.SolveConfig, start: Puzzle.State, **kwargs):
        search_result: SearchResult = SearchResult.build(
            statecls,
            batch_size,
            max_nodes,
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
        costs = jnp.zeros((batch_size,), dtype=KEY_DTYPE)
        hash_idxs = xnp.pad(hash_idx, (0, batch_size - 1))
        filled = jnp.zeros(batch_size, dtype=jnp.bool_).at[0].set(True)

        loop_state = LoopState(
            search_result=search_result,
            solve_config=solve_config,
            params=q_parameters,
            current=Current(hashidx=hash_idxs, cost=costs),
            filled=filled,
        )
        return loop_state

    def loop_condition(loop_state: LoopState):
        search_result = loop_state.search_result
        solve_config = loop_state.solve_config
        states = search_result.get_state(loop_state.current)
        filled = loop_state.filled
        hash_size = search_result.generated_size
        size_cond1 = filled.any()  # queue is not empty
        size_cond2 = hash_size < max_nodes  # hash table is not full
        size_cond = jnp.logical_and(size_cond1, size_cond2)

        solved = puzzle.batched_is_solved(solve_config, states)
        solved = jnp.logical_and(solved, filled)
        return jnp.logical_and(size_cond, ~solved.any())

    def loop_body(loop_state: LoopState):
        search_result = loop_state.search_result
        solve_config = loop_state.solve_config
        q_parameters = loop_state.params
        cost = loop_state.current.cost
        hash_idx = loop_state.current.hashidx
        filled = loop_state.filled
        states = search_result.get_state(loop_state.current)

        idx_tiles = xnp.tile(hash_idx, (puzzle.action_size, 1))  # [action_size, batch_size, ...]
        action = jnp.tile(
            jnp.arange(puzzle.action_size, dtype=ACTION_DTYPE)[:, jnp.newaxis],
            (1, cost.shape[0]),
        )  # [n_neighbours, batch_size]
        costs = jnp.tile(cost[jnp.newaxis, :], (puzzle.action_size, 1))  # [action_size, batch_size]
        filled_tiles = jnp.tile(
            filled[jnp.newaxis, :], (puzzle.action_size, 1)
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
                safe_old_dists = jnp.where(found, old_dists, -jnp.inf)
                dists = jnp.maximum(dists, safe_old_dists)
            else:
                safe_old_dists = jnp.where(found, old_dists, jnp.inf)
                dists = jnp.minimum(dists, safe_old_dists)

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

        flattened_neighbour_keys = jnp.where(optimal_mask, flattened_keys, jnp.inf)

        # Sort to keep best candidates
        sorted_key, sorted_idx = jax.lax.sort_key_val(
            flattened_neighbour_keys, jnp.arange(flattened_neighbour_keys.shape[-1])
        )
        sorted_vals = flattened_vals[sorted_idx]
        optimal_mask = optimal_mask[sorted_idx]

        neighbour_keys = sorted_key.reshape(puzzle.action_size, batch_size)
        vals = sorted_vals.reshape((puzzle.action_size, batch_size))
        optimal_mask = optimal_mask.reshape(puzzle.action_size, batch_size)

        def _insert(search_result: SearchResult, neighbour_keys, vals):

            search_result.priority_queue = search_result.priority_queue.insert(
                neighbour_keys,
                vals,
            )
            return search_result

        def _scan(search_result: SearchResult, val):
            neighbour_keys, vals, mask = val

            search_result = jax.lax.cond(
                jnp.any(mask),
                _insert,
                lambda search_result, *args: search_result,
                search_result,
                neighbour_keys,
                vals,
            )
            return search_result, None

        search_result, _ = jax.lax.scan(
            _scan,
            search_result,
            (neighbour_keys, vals, optimal_mask),
        )
        search_result, min_val, filled = search_result.pop_full_with_actions(puzzle, solve_config)

        return LoopState(
            search_result=search_result,
            solve_config=solve_config,
            params=q_parameters,
            current=min_val,
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
        states = search_result.get_state(current)

        solved = puzzle.batched_is_solved(solve_config, states)
        search_result.solved = solved.any()
        search_result.solved_idx = current[jnp.argmax(solved)]
        return search_result

    qstar_fn = jax.jit(qstar)
    empty_solve_config = puzzle.SolveConfig.default()
    empty_states = puzzle.State.default()

    if show_compile_time:
        print("initializing jit")
        start = time.time()

    # Pass empty states and target to JIT-compile the function with simple data.
    # Using actual puzzles would cause extremely long compilation times due to
    # tracing all possible functions. Empty inputs allow JAX to specialize the
    # compiled code without processing complex puzzle structures.
    qstar_fn(empty_solve_config, empty_states)

    if show_compile_time:
        end = time.time()
        print(f"Compile Time: {end - start:6.2f} seconds")
        print("JIT compiled\n\n")

    return qstar_fn
