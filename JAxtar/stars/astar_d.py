from typing import Any

import jax
import jax.numpy as jnp
import xtructure.numpy as xnp
from puxle import Puzzle

from helpers.jax_compile import compile_search_builder
from heuristic.heuristic_base import Heuristic
from JAxtar.annotate import KEY_DTYPE, MIN_BATCH_SIZE
from JAxtar.stars.search_base import (
    LoopStateWithStates,
    SearchResult,
    init_base_loop_state,
    base_loop_condition,
    build_deferred_loop_body,
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
        return init_base_loop_state(
            puzzle,
            search_result,
            solve_config,
            start,
            heuristic_parameters,
            search_result.batch_size,
        )

    def loop_condition(loop_state: LoopStateWithStates):
        return base_loop_condition(puzzle, loop_state, loop_state.states)

    def eval_fn(
        puzzle,
        search_result,
        solve_config,
        params,
        states,
        costs,
        filled_tiles,
        filled,
        look_ahead_pruning,
        cost_weight,
    ):
        action_size = search_result.action_size
        sr_batch_size = search_result.batch_size
        flattened_filled_tiles = filled_tiles.flatten()

        if look_ahead_pruning:
            neighbour_look_a_head, ncosts = puzzle.batched_get_neighbours(
                solve_config, states, filled
            )
            look_a_head_costs = costs + ncosts

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
            old_dists = search_result.get_dist(current_hash_idxs).reshape(
                action_size, sr_batch_size
            )

            need_compute = optimal_mask_reshaped & ~found_reshaped

            flat_states = neighbour_look_a_head.flatten()
            flat_need_compute = need_compute.flatten()

            # Clean duplicate sizes
            flat_size = action_size * sr_batch_size

            sorted_indices = stable_partition_three(
                flat_need_compute, jnp.zeros_like(flat_need_compute, dtype=jnp.bool_)
            )

            sorted_states = flat_states[sorted_indices]
            sorted_mask = flat_need_compute[sorted_indices]

            sorted_states_chunked = sorted_states.reshape((action_size, sr_batch_size))
            sorted_mask_chunked = sorted_mask.reshape((action_size, sr_batch_size))

            def _calc_heuristic_chunk(carry, input_slice):
                states_slice, compute_mask = input_slice
                h_val = variable_heuristic_batch_switcher(params, states_slice, compute_mask)
                return carry, h_val

            _, h_val_chunks = jax.lax.scan(
                _calc_heuristic_chunk,
                None,
                (sorted_states_chunked, sorted_mask_chunked),
            )

            h_val_sorted = h_val_chunks.reshape(-1)
            flat_h_val = (
                jnp.empty((flat_size,), dtype=h_val_sorted.dtype)
                .at[sorted_indices]
                .set(h_val_sorted)
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
            heuristic_vals = variable_heuristic_batch_switcher(params, states, filled)
            heuristic_vals = jnp.where(filled, heuristic_vals, jnp.inf)
            heuristic_vals = jnp.tile(heuristic_vals[jnp.newaxis, :], (action_size, 1)).astype(
                KEY_DTYPE
            )

            optimal_mask = flattened_filled_tiles.reshape(action_size, sr_batch_size)
            neighbour_keys = (cost_weight * costs + heuristic_vals).astype(KEY_DTYPE)

        return neighbour_keys, heuristic_vals, optimal_mask

    loop_body = build_deferred_loop_body(
        puzzle, look_ahead_pruning, cost_weight, eval_fn, use_heuristic=True
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
