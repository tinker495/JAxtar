from typing import Any

import jax
import jax.numpy as jnp
import xtructure.numpy as xnp
from puxle import Puzzle

from helpers.jax_compile import compile_search_builder
from heuristic.heuristic_base import Heuristic
from JAxtar.annotate import ACTION_DTYPE, KEY_DTYPE, MIN_BATCH_SIZE
from JAxtar.stars.search_base import (
    Current,
    LoopStateWithStates,
    Parant_with_Costs,
    Parent,
    SearchResult,
    insert_priority_queue_batches,
)
from JAxtar.utils.array_ops import stable_partition_three
from JAxtar.utils.batch_switcher import variable_batch_switcher_builder


def _astar_d_loop_builder(
    puzzle: Puzzle,
    heuristic: Heuristic,
    batch_size: int = 1024,
    max_nodes: int = int(1e6),
    pop_ratio: float = jnp.inf,
    cost_weight: float = 1.0 - 1e-6,
    look_ahead_pruning: bool = True,
):
    # Loop builder isolates init/cond/body so downstream code can tap
    # into intermediate loop data (params, queue state) without
    # duplicating the search wiring or triggering extra traces.
    statecls = puzzle.State
    action_size = puzzle.action_size

    variable_heuristic_batch_switcher = variable_batch_switcher_builder(
        heuristic.batched_distance,
        max_batch_size=batch_size,
        min_batch_size=MIN_BATCH_SIZE,
        pad_value=jnp.inf,
    )
    denom = max(1, puzzle.action_size // 2)
    min_pop = max(1, MIN_BATCH_SIZE // denom)

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
        heuristic_parameters = heuristic.prepare_heuristic_parameters(solve_config, **kwargs)

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

        return LoopStateWithStates(
            search_result=search_result,
            solve_config=solve_config,
            params=heuristic_parameters,
            current=Current(hashidx=hash_idxs, cost=costs),
            states=states,
            filled=filled,
        )

    def loop_condition(loop_state: LoopStateWithStates):
        search_result = loop_state.search_result
        solve_config = loop_state.solve_config
        states = loop_state.states
        filled = loop_state.filled
        hash_size = search_result.generated_size
        size_cond1 = filled.any()  # queue is not empty
        size_cond2 = hash_size < search_result.capacity  # hash table is not full
        size_cond = jnp.logical_and(size_cond1, size_cond2)

        solved = puzzle.batched_is_solved(solve_config, states)
        solved = jnp.logical_and(solved, filled)
        return jnp.logical_and(size_cond, ~solved.any())

    def loop_body(loop_state: LoopStateWithStates):
        search_result = loop_state.search_result
        solve_config = loop_state.solve_config
        heuristic_parameters = loop_state.params
        cost = loop_state.current.cost
        hash_idx = loop_state.current.hashidx
        filled = loop_state.filled
        states = loop_state.states

        action_size = search_result.action_size
        sr_batch_size = search_result.batch_size
        flat_size = action_size * sr_batch_size
        idx_tiles = xnp.tile(hash_idx, (action_size, 1))  # [action_size, batch_size, ...]
        action = jnp.tile(
            jnp.arange(action_size, dtype=ACTION_DTYPE)[:, jnp.newaxis],
            (1, sr_batch_size),
        )  # [n_neighbours, batch_size]
        costs = jnp.tile(cost[jnp.newaxis, :], (action_size, 1))  # [action_size, batch_size]
        filled_tiles = jnp.tile(
            filled[jnp.newaxis, :], (action_size, 1)
        )  # [action_size, batch_size]

        flattened_filled_tiles = filled_tiles.flatten()

        if look_ahead_pruning:
            neighbour_look_a_head, ncosts = puzzle.batched_get_neighbours(
                solve_config, states, filled
            )  # [action_size, batch_size]
            look_a_head_costs = costs + ncosts  # [action_size, batch_size]

            flattened_neighbour_look_head = neighbour_look_a_head.flatten()
            flattened_look_a_head_costs = look_a_head_costs.flatten()

            current_hash_idxs, found = search_result.hashtable.lookup_parallel(
                flattened_neighbour_look_head, flattened_filled_tiles
            )

            old_costs = search_result.get_cost(current_hash_idxs)

            candidate_mask = jnp.logical_or(
                ~found, jnp.less(flattened_look_a_head_costs, old_costs)
            )
            candidate_mask = candidate_mask & flattened_filled_tiles

            optimal_mask = (
                xnp.unique_mask(
                    flattened_neighbour_look_head, flattened_look_a_head_costs, candidate_mask
                )
                & candidate_mask
            )

            found_reshaped = found.reshape(action_size, sr_batch_size)
            optimal_mask_reshaped = optimal_mask.reshape(action_size, sr_batch_size)

            old_dists = search_result.get_dist(current_hash_idxs)  # flattened
            old_dists = old_dists.reshape(action_size, sr_batch_size)

            need_compute = optimal_mask_reshaped & ~found_reshaped

            # Pack all `need_compute=True` entries contiguously across the full
            # (action_size * batch_size) batch to maximize effective batch size
            # in `variable_heuristic_batch_switcher` (less padding / fewer small calls).
            flat_states = neighbour_look_a_head.flatten()
            flat_need_compute = need_compute.flatten()

            n = flat_size
            n = flat_size
            # Stable sort so `need_compute=True` comes first (key False), preserving order.
            # Stable sort so `need_compute=True` comes first (key False), preserving order.
            sorted_indices = stable_partition_three(
                flat_need_compute, jnp.zeros_like(flat_need_compute, dtype=jnp.bool_)
            )

            sorted_states = flat_states[sorted_indices]
            sorted_mask = flat_need_compute[sorted_indices]

            # `variable_heuristic_batch_switcher` is built with max_batch_size=batch_size,
            # so we must not call it with a larger leading dimension than `batch_size`.
            # Reshape the globally-packed vector into `action_size` chunks of `batch_size`
            # and compute per-chunk via scan.
            # `sorted_states` is a (flattened) Puzzle.State pytree (xtructure),
            # so reshape expects just the leading batch shape.
            sorted_states_chunked = sorted_states.reshape((action_size, sr_batch_size))
            sorted_mask_chunked = sorted_mask.reshape((action_size, sr_batch_size))

            def _calc_heuristic_chunk(carry, input_slice):
                states_slice, compute_mask = input_slice
                h_val = variable_heuristic_batch_switcher(
                    heuristic_parameters, states_slice, compute_mask
                )
                return carry, h_val

            _, h_val_chunks = jax.lax.scan(
                _calc_heuristic_chunk,
                None,
                (sorted_states_chunked, sorted_mask_chunked),
            )  # [action_size, batch_size]

            h_val_sorted = h_val_chunks.reshape(-1)  # [action_size * batch_size]
            flat_h_val = (
                jnp.empty((n,), dtype=h_val_sorted.dtype).at[sorted_indices].set(h_val_sorted)
            )
            computed_heuristic_vals = flat_h_val.reshape(action_size, sr_batch_size)

            heuristic_vals = jnp.where(
                found_reshaped,
                old_dists,
                computed_heuristic_vals,
            )
            heuristic_vals = jnp.where(filled_tiles, heuristic_vals, jnp.inf).astype(KEY_DTYPE)

            neighbour_keys = (cost_weight * look_a_head_costs + heuristic_vals).astype(KEY_DTYPE)

        else:

            heuristic_vals = variable_heuristic_batch_switcher(
                heuristic_parameters, states, filled
            )  # [batch_size]
            heuristic_vals = jnp.where(filled, heuristic_vals, jnp.inf)  # [batch_size]
            heuristic_vals = jnp.tile(heuristic_vals[jnp.newaxis, :], (action_size, 1)).astype(
                KEY_DTYPE
            )  # [action_size, batch_size]

            optimal_mask = flattened_filled_tiles

            neighbour_keys = (cost_weight * costs + heuristic_vals).astype(KEY_DTYPE)

        vals = Parant_with_Costs(
            parent=Parent(hashidx=idx_tiles.flatten(), action=action.flatten()),
            cost=costs.flatten(),
            dist=heuristic_vals.flatten(),
        )
        flattened_vals = vals.flatten()
        flattened_keys = neighbour_keys.flatten()

        flattened_neighbour_keys = jnp.where(optimal_mask, flattened_keys, jnp.inf)

        # Sort to keep best candidates
        sorted_key, sorted_idx = jax.lax.sort_key_val(
            flattened_neighbour_keys, jnp.arange(flat_size)
        )
        sorted_vals = flattened_vals[sorted_idx]
        sorted_optimal_unique_mask = optimal_mask[sorted_idx]

        neighbour_keys = sorted_key.reshape(action_size, sr_batch_size)
        vals = sorted_vals.reshape((action_size, sr_batch_size))
        optimal_unique_mask = sorted_optimal_unique_mask.reshape(action_size, sr_batch_size)

        search_result = insert_priority_queue_batches(
            search_result,
            neighbour_keys,
            vals,
            optimal_unique_mask,
        )
        search_result, min_val, next_states, filled = search_result.pop_full_with_actions(
            puzzle=puzzle,
            solve_config=solve_config,
            use_heuristic=True,
        )

        return LoopStateWithStates(
            search_result=search_result,
            solve_config=solve_config,
            params=heuristic_parameters,
            current=min_val,
            states=next_states,
            filled=filled,
        )

    return init_loop_state, loop_condition, loop_body


def astar_d_builder(
    puzzle: Puzzle,
    heuristic: Heuristic,
    batch_size: int = 1024,
    max_nodes: int = int(1e6),
    pop_ratio: float = jnp.inf,
    cost_weight: float = 1.0 - 1e-6,
    show_compile_time: bool = False,
    look_ahead_pruning: bool = True,
    warmup_inputs: tuple[Puzzle.SolveConfig, Puzzle.State] | None = None,
):
    """
    Builds and returns a JAX-accelerated A* with deferred node evaluation (A* deferred).

    In standard A*, when a node is expanded, all its children are generated and their heuristics
    are evaluated immediately to calculate f(n) = g(n) + h(n) for insertion into the open list.

    In A* deferred, we delay the generation and heuristic evaluation of children. Instead,
    when a node is expanded, we insert its potential *actions* (edges) into the priority queue,
    using the *parent's* f-value (or similar) as the priority.
    Only when an action is popped from the queue do we actually generate the child state
    and evaluate its heuristic (if it becomes a parent for the next step).

    This approach is beneficial when:
    1. The heuristic calculation is expensive.
    2. The branching factor is large (avoids evaluating children that are never explored).
    3. We want to maximize batch efficiency in JAX by keeping the pipeline uniform.

    Args:
        puzzle: Puzzle instance that defines the problem space and operations.
        heuristic: Heuristic instance that provides state evaluation.
        batch_size: Number of states to process in parallel (default: 1024).
        max_nodes: Maximum number of nodes to explore before terminating (default: 1e6).
        pop_ratio: Ratio of states to pop from the priority queue.
        cost_weight: Weight applied to the path cost in the A* with deferred search algorithm (default: 1.0-1e-6).
                    Values closer to 1.0 make the search more greedy/depth-first.
        show_compile_time: If True, displays the time taken to compile the search function (default: False).
        look_ahead_pruning: Enables neighbour look-ahead pruning (default: True). Disable
            to recover canonical A* deferred behaviour when simulator cost outweighs queue
            pressure.

    Returns:
        A function that performs A* with deferred search given a start state and solve configuration.
    """

    init_loop_state, loop_condition, loop_body = _astar_d_loop_builder(
        puzzle, heuristic, batch_size, max_nodes, pop_ratio, cost_weight, look_ahead_pruning
    )

    def astar_d(
        solve_config: Puzzle.SolveConfig,
        start: Puzzle.State,
        **kwargs: Any,
    ) -> SearchResult:
        """
        astar_d is the implementation of the A* with deferred search algorithm.
        """
        loop_state = init_loop_state(solve_config, start, **kwargs)
        loop_state = jax.lax.while_loop(loop_condition, loop_body, loop_state)
        search_result = loop_state.search_result
        current = loop_state.current
        states = loop_state.states
        filled = loop_state.filled
        solved = puzzle.batched_is_solved(solve_config, states)
        solved = jnp.logical_and(solved, filled)
        search_result.solved = solved.any()
        search_result.solved_idx = current[jnp.argmax(solved)]
        return search_result

    return compile_search_builder(astar_d, puzzle, show_compile_time, warmup_inputs)
