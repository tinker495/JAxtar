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
    LoopState,
    Parent,
    SearchResult,
    insert_priority_queue_batches,
    init_base_loop_state_current,
    base_loop_condition_current,
)
from JAxtar.utils.array_ops import stable_partition_three
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
        return init_base_loop_state_current(
            puzzle,
            search_result,
            solve_config,
            start,
            heuristic_parameters,
            search_result.batch_size,
        )

    def loop_condition(loop_state: LoopState):
        return base_loop_condition_current(puzzle, loop_state)

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
        parent_action = jnp.tile(
            jnp.arange(action_size, dtype=ACTION_DTYPE)[:, jnp.newaxis],
            (1, sr_batch_size),
        )  # [n_neighbours, batch_size]
        nextcosts = (current.cost[jnp.newaxis, :] + ncost).astype(
            KEY_DTYPE
        )  # [n_neighbours, batch_size]
        filleds = jnp.isfinite(nextcosts)  # [n_neighbours, batch_size]
        # Use int32 for indexing; ACTION_DTYPE (uint8) overflows when batch_size > 255.
        parent_index = jnp.tile(
            jnp.arange(sr_batch_size, dtype=jnp.int32)[jnp.newaxis, :],
            (action_size, 1),
        )  # [n_neighbours, batch_size]
        unflatten_shape = (action_size, sr_batch_size)

        parent = Parent(
            hashidx=current.hashidx[parent_index],
            action=parent_action,
        )

        flatten_neighbours = neighbours.flatten()
        flatten_filleds = filleds.flatten()
        flatten_nextcosts = nextcosts.flatten()
        flatten_parents = parent.flatten()

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
        # Stable partition to group useful entries first.
        # Improves computational efficiency by gathering only batches with samples that need updates.
        invperm = stable_partition_three(flatten_new_states_mask, final_process_mask)

        flatten_final_process_mask = final_process_mask[invperm]
        flatten_new_states_mask = flatten_new_states_mask[invperm]
        flatten_neighbours = flatten_neighbours[invperm]
        flatten_nextcosts = flatten_nextcosts[invperm]

        hash_idx = hash_idx[invperm]
        vals = Current(hashidx=hash_idx, cost=flatten_nextcosts).reshape(unflatten_shape)
        neighbours = flatten_neighbours.reshape(unflatten_shape)
        new_states_mask = flatten_new_states_mask.reshape(unflatten_shape)
        final_process_mask = flatten_final_process_mask.reshape(unflatten_shape)

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
        solved = puzzle.batched_is_solved(solve_config, states)
        solved = jnp.logical_and(solved, filled)
        search_result.solved = solved.any()
        search_result.solved_idx = current[jnp.argmax(solved)]
        return search_result

    return compile_search_builder(astar, puzzle, show_compile_time, warmup_inputs)
