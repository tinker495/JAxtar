from typing import Any

import jax
import jax.numpy as jnp
import xtructure.numpy as xnp
from puxle import Puzzle

from helpers.jax_compile import compile_search_builder
from JAxtar.annotate import KEY_DTYPE, MIN_BATCH_SIZE
from JAxtar.stars.search_base import (
    LoopStateWithStates,
    SearchResult,
    init_base_loop_state,
    base_loop_condition,
    build_deferred_loop_body,
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
    statecls = puzzle.State
    action_size = puzzle.action_size

    dist_sign = -1.0 if pessimistic_update else 1.0
    denom = max(1, puzzle.action_size // 2)
    min_pop = max(1, MIN_BATCH_SIZE // denom)

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
        return init_base_loop_state(
            puzzle, search_result, solve_config, start, q_parameters, search_result.batch_size
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

        q_vals = variable_q_batch_switcher(params, states, filled)
        q_vals = q_vals.transpose().astype(KEY_DTYPE)

        neighbour_keys = (cost_weight * costs + q_vals).astype(KEY_DTYPE)
        neighbour_keys = jnp.where(filled_tiles, neighbour_keys, jnp.inf)

        dists = q_vals.flatten()

        if look_ahead_pruning:
            neighbour_look_a_head, ncosts = puzzle.batched_get_neighbours(
                solve_config, states, filled
            )
            look_a_head_costs = costs + ncosts
            flattened_neighbour_look_head = neighbour_look_a_head.flatten()
            flattened_look_a_head_costs = look_a_head_costs.flatten()

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

            if pessimistic_update:
                step_cost = ncosts.flatten().astype(KEY_DTYPE)
                q_old = old_dists.astype(KEY_DTYPE) + step_cost
                q_old_for_max = jnp.where(found, q_old, -jnp.inf)
                dists = jnp.maximum(dists, q_old_for_max)
            else:
                step_cost = ncosts.flatten().astype(KEY_DTYPE)
                q_old = old_dists.astype(KEY_DTYPE) + step_cost
                q_old_for_min = jnp.where(found, q_old, jnp.inf)
                dists = jnp.minimum(dists, q_old_for_min)

            better_cost_mask = jnp.less(flattened_look_a_head_costs, old_costs)
            optimal_mask = unique_mask & (jnp.logical_or(~found, better_cost_mask))
        else:
            optimal_mask = flattened_filled_tiles

        return (
            neighbour_keys,
            dists.reshape(action_size, sr_batch_size),
            optimal_mask.reshape(action_size, sr_batch_size),
        )

    loop_body = build_deferred_loop_body(
        puzzle, look_ahead_pruning, cost_weight, eval_fn, use_heuristic=False
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

        solved = puzzle.batched_is_solved(solve_config, states)
        solved = jnp.logical_and(solved, filled)
        search_result.solved = solved.any()
        search_result.solved_idx = current[jnp.argmax(solved)]
        return search_result

    return compile_search_builder(qstar, puzzle, show_compile_time, warmup_inputs)
