from typing import Any

import jax
import jax.numpy as jnp
import xtructure.numpy as xnp
from puxle import Puzzle

from helpers.jax_compile import jit_with_warmup
from heuristic.heuristic_base import Heuristic
from JAxtar.annotate import KEY_DTYPE, MIN_BATCH_SIZE
from JAxtar.stars.search_base import (
    Current,
    LoopState,
    Parent,
    SearchResult,
    build_action_major_parent_layout,
    finalize_search_result,
    insert_priority_queue_batches,
    loop_continue_if_not_solved,
    partition_and_pack_frontier_candidates,
)
from JAxtar.utils.batch_switcher import variable_batch_switcher_builder


def _astar_loop_builder(
    puzzle: Puzzle,
    heuristic: Heuristic,
    batch_size: int = 1024,
    max_nodes: int = int(1e6),
    pop_ratio: float = jnp.inf,
    cost_weight: float = 1.0 - 1e-6,
):
    # The loop builder factors out loop init/condition/body so callers
    # can reuse intermediate loop data (e.g., parameters, queue state)
    # without retracing or reassembling the search plumbing each time.
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
        )
        heuristic_parameters = heuristic.prepare_heuristic_parameters(solve_config, **kwargs)

        (
            search_result.hashtable,
            _,
            hash_idx,
        ) = search_result.hashtable.insert(start)

        search_result.cost = search_result.cost.at[hash_idx.index].set(0)
        sr_batch_size = search_result.batch_size
        hash_idxs = xnp.pad(hash_idx, (0, sr_batch_size - 1))
        costs = jnp.full((sr_batch_size,), jnp.inf, dtype=KEY_DTYPE).at[0].set(0)
        filled = jnp.zeros(sr_batch_size, dtype=jnp.bool_).at[0].set(True)

        return LoopState(
            search_result=search_result,
            solve_config=solve_config,
            params=heuristic_parameters,
            current=Current(hashidx=hash_idxs, cost=costs),
            filled=filled,
        )

    def loop_condition(loop_state: LoopState):
        search_result = loop_state.search_result
        solve_config = loop_state.solve_config
        states = search_result.get_state(loop_state.current)
        filled = loop_state.filled
        return loop_continue_if_not_solved(search_result, puzzle, solve_config, states, filled)

    def loop_body(loop_state: LoopState):
        search_result = loop_state.search_result
        solve_config = loop_state.solve_config
        heuristic_parameters = loop_state.params
        current = loop_state.current
        filled = loop_state.filled
        states = search_result.get_state(current)

        neighbours, ncost = puzzle.batched_get_neighbours(solve_config, states, filled)
        action_size = search_result.action_size
        sr_batch_size = search_result.batch_size
        flat_parent_indices, flat_parent_actions, _ = build_action_major_parent_layout(
            action_size, sr_batch_size
        )
        nextcosts = (current.cost[jnp.newaxis, :] + ncost).astype(
            KEY_DTYPE
        )  # [n_neighbours, batch_size]
        filleds = jnp.isfinite(nextcosts)  # [n_neighbours, batch_size]

        parent = Parent(
            hashidx=current.hashidx[flat_parent_indices],
            action=flat_parent_actions,
        )

        flatten_neighbours = neighbours.flatten()
        flatten_filleds = filleds.flatten()
        flatten_nextcosts = nextcosts.flatten()
        flatten_parents = parent

        (
            search_result.hashtable,
            flatten_new_states_mask,
            cheapest_uniques_mask,
            hash_idx,
        ) = search_result.hashtable.parallel_insert(
            flatten_neighbours, flatten_filleds, flatten_nextcosts
        )

        # It must also be cheaper than any previously found path to this state.
        optimal_mask = jnp.less(flatten_nextcosts, search_result.get_cost(hash_idx))

        # Combine all conditions for the final decision.
        final_process_mask = jnp.logical_and(cheapest_uniques_mask, optimal_mask)

        # Update the cost (g-value) for the newly found optimal paths before they are
        # masked out. This ensures the cost table is always up-to-date.
        search_result.cost = xnp.update_on_condition(
            search_result.cost,
            hash_idx.index,
            final_process_mask,
            flatten_nextcosts,  # Use costs before they are set to inf
        )
        search_result.parent = xnp.update_on_condition(
            search_result.parent,
            hash_idx.index,
            final_process_mask,
            flatten_parents,
        )

        # Apply the final mask: deactivate non-optimal nodes by setting their cost to infinity
        # and updating the insertion flag. This ensures they are ignored in subsequent steps.
        flatten_nextcosts = jnp.where(final_process_mask, flatten_nextcosts, jnp.inf)
        (
            vals,
            neighbours,
            new_states_mask,
            final_process_mask,
        ) = partition_and_pack_frontier_candidates(
            flatten_new_states_mask,
            final_process_mask,
            flatten_neighbours,
            flatten_nextcosts,
            hash_idx,
            action_size,
            sr_batch_size,
        )

        def _new_states(search_result: SearchResult, vals, neighbour, new_states_mask):
            neighbour_heur = variable_heuristic_batch_switcher(
                heuristic_parameters, neighbour, new_states_mask
            ).astype(KEY_DTYPE)
            # cache the heuristic value
            search_result.dist = xnp.update_on_condition(
                search_result.dist,
                vals.hashidx.index,
                new_states_mask,
                neighbour_heur,
            )
            return search_result, neighbour_heur

        def _old_states(search_result: SearchResult, vals, neighbour, new_states_mask):
            neighbour_heur = search_result.dist[vals.hashidx.index]
            return search_result, neighbour_heur

        def _scan(search_result: SearchResult, val):
            vals, neighbour, new_states_mask = val

            search_result, neighbour_heur = jax.lax.cond(
                jnp.any(new_states_mask),
                _new_states,
                _old_states,
                search_result,
                vals,
                neighbour,
                new_states_mask,
            )

            neighbour_key = (cost_weight * vals.cost + neighbour_heur).astype(KEY_DTYPE)
            return search_result, neighbour_key

        search_result, neighbour_keys = jax.lax.scan(
            _scan,
            search_result,
            (vals, neighbours, new_states_mask),
        )
        search_result = insert_priority_queue_batches(
            search_result,
            neighbour_keys,
            vals,
            final_process_mask,
        )
        search_result, current, filled = search_result.pop_full()
        return LoopState(
            search_result=search_result,
            solve_config=solve_config,
            params=heuristic_parameters,
            current=current,
            filled=filled,
        )

    return init_loop_state, loop_condition, loop_body


def astar_builder(
    puzzle: Puzzle,
    heuristic: Heuristic,
    batch_size: int = 1024,
    max_nodes: int = int(1e6),
    pop_ratio: float = jnp.inf,
    cost_weight: float = 1.0 - 1e-6,
    show_compile_time: bool = False,
    warmup_inputs: tuple[Puzzle.SolveConfig, Puzzle.State] | None = None,
):
    """
    Builds and returns a JAX-accelerated A* search function.

    Args:
        puzzle: Puzzle instance that defines the problem space and operations.
        heuristic: Heuristic instance that provides state evaluation.
        batch_size: Number of states to process in parallel (default: 1024).
        max_nodes: Maximum number of nodes to explore before terminating (default: 1e6).
        cost_weight: Weight applied to the path cost in f(n) = g(n) + w*h(n) (default: 1.0-1e-6).
                    Values closer to 1.0 make the search more greedy/depth-first.
        show_compile_time: If True, displays the time taken to compile the search function (default: False).

    Returns:
        A function that performs A* search given a start state and solve configuration.
    """

    init_loop_state, loop_condition, loop_body = _astar_loop_builder(
        puzzle, heuristic, batch_size, max_nodes, pop_ratio, cost_weight
    )

    def astar(
        solve_config: Puzzle.SolveConfig,
        start: Puzzle.State,
        **kwargs: Any,
    ) -> SearchResult:
        """
        astar is the implementation of the A* algorithm.
        """
        loop_state = init_loop_state(solve_config, start, **kwargs)
        loop_state = jax.lax.while_loop(loop_condition, loop_body, loop_state)
        search_result = loop_state.search_result
        current = loop_state.current
        filled = loop_state.filled
        states = search_result.get_state(current)
        solved_mask = jnp.logical_and(puzzle.batched_is_solved(solve_config, states), filled)
        return finalize_search_result(search_result, current, solved_mask)

    return jit_with_warmup(
        astar,
        puzzle=puzzle,
        show_compile_time=show_compile_time,
        warmup_inputs=warmup_inputs,
    )
