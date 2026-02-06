"""JAxtar Bidirectional A* Search Implementation.

This module implements bidirectional A* search, which explores from both start
and goal states simultaneously.

Notes on optimality:
- By default (`terminate_on_first_solution=True` in `bi_astar_builder()`), the
  loop terminates as soon as *any* meeting point is found. This is fast, but it
  is not an optimality proof.
- If you need an optimality proof, set `terminate_on_first_solution=False` and
  ensure the heuristic semantics used in PQ keys satisfy the usual A* conditions
  for the chosen domain.

Key Benefits:
- Reduces search space from O(b^d) to approximately O(b^(d/2))
- Particularly effective for puzzles with symmetric forward/backward transitions
"""

import time
from typing import Any

import chex
import jax
import jax.numpy as jnp
import xtructure.numpy as xnp
from puxle import Puzzle

from helpers.jax_compile import compile_with_example
from heuristic.heuristic_base import Heuristic
from JAxtar.annotate import ACTION_DTYPE, KEY_DTYPE, MIN_BATCH_UNIT
from JAxtar.bi_stars.bi_search_base import (
    BiDirectionalSearchResult,
    BiLoopState,
    build_bi_search_result,
    check_intersection,
    common_bi_loop_condition,
    initialize_bi_loop_common,
    update_meeting_point,
)
from JAxtar.stars.search_base import Current, Parent, SearchResult
from JAxtar.utils.array_ops import stable_partition_three
from JAxtar.utils.batch_switcher import variable_batch_switcher_builder


def _bi_astar_loop_builder(
    puzzle: Puzzle,
    heuristic: Heuristic,
    bi_result_template: BiDirectionalSearchResult,
    batch_size: int = 1024,
    cost_weight: float = 1.0 - 1e-6,
    use_backward_heuristic: bool = True,
    terminate_on_first_solution: bool = False,
):
    """
    Build the loop components for bidirectional A* search.

    Args:
        puzzle: Puzzle instance
        heuristic: Heuristic instance (used for both directions)
        bi_result_template: Pre-built BiDirectionalSearchResult template
        batch_size: Batch size for parallel processing
        cost_weight: Weight for path cost in f = cost_weight * g + h

    Returns:
        Tuple of (init_loop_state, loop_condition, loop_body) functions
    """
    action_size = puzzle.action_size

    variable_heuristic_batch_switcher = variable_batch_switcher_builder(
        heuristic.batched_distance,
        pad_value=jnp.inf,
    )

    def init_loop_state(
        bi_result: BiDirectionalSearchResult,
        solve_config: Puzzle.SolveConfig,
        inverse_solveconfig: Puzzle.SolveConfig,
        start: Puzzle.State,
        heuristic_params_forward: Any,
        heuristic_params_backward: Any,
    ) -> BiLoopState:
        """Initialize bidirectional search from start and goal states."""

        (
            fwd_filled,
            fwd_current,
            _,
            bwd_filled,
            bwd_current,
            _,
        ) = initialize_bi_loop_common(bi_result, puzzle, solve_config, start)

        return BiLoopState(
            bi_result=bi_result,
            solve_config=solve_config,
            inverse_solveconfig=inverse_solveconfig,
            params_forward=heuristic_params_forward,
            params_backward=heuristic_params_backward,
            current_forward=fwd_current,
            current_backward=bwd_current,
            filled_forward=fwd_filled,
            filled_backward=bwd_filled,
        )

    def loop_condition(loop_state: BiLoopState) -> chex.Array:
        """
        Check if search should continue.

        Continues while:
        1. At least one direction can still expand nodes (has frontier AND hashtable capacity)
        2. Termination condition not met (lower_bound < upper_bound)
        """
        return common_bi_loop_condition(
            loop_state.bi_result,
            loop_state.filled_forward,
            loop_state.filled_backward,
            loop_state.current_forward,
            loop_state.current_backward,
            cost_weight,
            terminate_on_first_solution,
        )

    def _expand_direction(
        bi_result: BiDirectionalSearchResult,
        solve_config: Puzzle.SolveConfig,
        inverse_solveconfig: Puzzle.SolveConfig,
        heuristic_params: Any,
        current: Current,
        filled: chex.Array,
        is_forward: bool,
        use_heuristic: bool,
    ) -> tuple[BiDirectionalSearchResult, Current, chex.Array]:
        """
        Expand one direction (forward or backward).

        Args:
            bi_result: Current bidirectional search result
            solve_config: Puzzle solve configuration
            heuristic_params: Parameters for heuristic evaluation
            current: Current batch of nodes to expand
            filled: Boolean mask for valid entries
            is_forward: True for forward expansion, False for backward

        Returns:
            Updated (bi_result, current, filled)
        """
        if is_forward:
            search_result = bi_result.forward
            opposite_sr = bi_result.backward
            current_solve_config = solve_config
            get_neighbours_fn = puzzle.batched_get_neighbours
        else:
            search_result = bi_result.backward
            opposite_sr = bi_result.forward
            current_solve_config = inverse_solveconfig
            get_neighbours_fn = puzzle.batched_get_inverse_neighbours

        sr_batch_size = search_result.batch_size

        # Get states for current batch
        states = search_result.get_state(current)

        # Get neighbors
        neighbours, ncost = get_neighbours_fn(current_solve_config, states, filled)
        # neighbours: [action_size, batch_size]
        # ncost: [action_size, batch_size]

        # Prepare parent information
        parent_action = jnp.tile(
            jnp.arange(action_size, dtype=ACTION_DTYPE)[:, jnp.newaxis],
            (1, sr_batch_size),
        )  # [action_size, batch_size]
        nextcosts = (current.cost[jnp.newaxis, :] + ncost).astype(KEY_DTYPE)
        filleds = jnp.isfinite(nextcosts)  # [action_size, batch_size]
        parent_index = jnp.tile(
            jnp.arange(sr_batch_size, dtype=jnp.int32)[jnp.newaxis, :],
            (action_size, 1),
        )
        unflatten_shape = (action_size, sr_batch_size)

        parent = Parent(
            hashidx=current.hashidx[parent_index],
            action=parent_action,
        )

        # Flatten for batch processing
        flatten_neighbours = neighbours.flatten()
        flatten_filleds = filleds.flatten()
        flatten_nextcosts = nextcosts.flatten()
        flatten_parents = parent.flatten()

        # Insert into hash table
        (
            search_result.hashtable,
            flatten_new_states_mask,
            cheapest_uniques_mask,
            hash_idx,
        ) = search_result.hashtable.parallel_insert(
            flatten_neighbours, flatten_filleds, flatten_nextcosts
        )

        # Check optimality
        optimal_mask = jnp.less(flatten_nextcosts, search_result.get_cost(hash_idx))
        final_process_mask = jnp.logical_and(cheapest_uniques_mask, optimal_mask)

        # Update cost and parent
        search_result.cost = xnp.update_on_condition(
            search_result.cost,
            hash_idx.index,
            final_process_mask,
            flatten_nextcosts,
        )
        search_result.parent = xnp.update_on_condition(
            search_result.parent,
            hash_idx.index,
            final_process_mask,
            flatten_parents,
        )

        # Check intersection with opposite frontier
        # IMPORTANT: Use flatten_filleds (all valid neighbors), not final_process_mask
        # We need to check ALL valid expanded states against the opposite hash table,
        # not just the ones that were newly inserted with optimal costs.
        # This is because a state might already exist in our hash table but
        # was just added to the opposite hash table in the current iteration.
        found_mask, opposite_hashidx, opposite_costs, _ = check_intersection(
            flatten_neighbours,
            flatten_nextcosts,
            flatten_filleds,
            opposite_sr,
        )

        # Update meeting point
        # IMPORTANT: use the g-values stored in this direction's hash table.
        # `flatten_nextcosts` is the candidate expansion cost and can differ from the
        # stored (cheapest known) g-value due to duplicates / re-discovery.
        this_costs = search_result.get_cost(hash_idx)
        total_costs = this_costs + opposite_costs
        bi_result.meeting = update_meeting_point(
            bi_result.meeting,
            found_mask,
            hash_idx,
            opposite_hashidx,
            this_costs,
            opposite_costs,
            total_costs,
            is_forward,
        )

        # Stable partition for efficiency
        invperm = stable_partition_three(flatten_new_states_mask, final_process_mask)
        flatten_final_process_mask = final_process_mask[invperm]
        flatten_new_states_mask = flatten_new_states_mask[invperm]
        flatten_neighbours = flatten_neighbours[invperm]
        flatten_nextcosts = jnp.where(final_process_mask, flatten_nextcosts, jnp.inf)[invperm]
        hash_idx = hash_idx[invperm]

        vals = Current(hashidx=hash_idx, cost=flatten_nextcosts).reshape(unflatten_shape)
        neighbours_reshaped = flatten_neighbours.reshape(unflatten_shape)
        new_states_mask = flatten_new_states_mask.reshape(unflatten_shape)
        final_process_mask_reshaped = flatten_final_process_mask.reshape(unflatten_shape)

        # Compute heuristic and insert into priority queue
        def _new_states(sr: SearchResult, vals, neighbour, new_states_mask):
            if use_heuristic:
                neighbour_heur = variable_heuristic_batch_switcher(
                    heuristic_params, neighbour, new_states_mask
                ).astype(KEY_DTYPE)
            else:
                neighbour_heur = jnp.zeros_like(vals.cost, dtype=KEY_DTYPE)
            sr.dist = xnp.update_on_condition(
                sr.dist,
                vals.hashidx.index,
                new_states_mask,
                neighbour_heur,
            )
            return sr, neighbour_heur

        def _old_states(sr: SearchResult, vals, neighbour, new_states_mask):
            neighbour_heur = sr.dist[vals.hashidx.index]
            return sr, neighbour_heur

        def _inserted(sr: SearchResult, vals, neighbour_heur):
            neighbour_key = (cost_weight * vals.cost + neighbour_heur).astype(KEY_DTYPE)
            sr.priority_queue = sr.priority_queue.insert(neighbour_key, vals)
            return sr

        def _scan(sr: SearchResult, val):
            vals, neighbour, new_states_mask, final_process_mask = val
            sr, neighbour_heur = jax.lax.cond(
                jnp.any(new_states_mask),
                _new_states,
                _old_states,
                sr,
                vals,
                neighbour,
                new_states_mask,
            )
            sr = jax.lax.cond(
                jnp.any(final_process_mask),
                _inserted,
                lambda sr, *args: sr,
                sr,
                vals,
                neighbour_heur,
            )
            return sr, None

        search_result, _ = jax.lax.scan(
            _scan,
            search_result,
            (vals, neighbours_reshaped, new_states_mask, final_process_mask_reshaped),
        )

        # Pop next batch
        search_result, new_current, new_filled = search_result.pop_full()

        # Update bi_result with modified search_result
        if is_forward:
            bi_result.forward = search_result
        else:
            bi_result.backward = search_result

        return bi_result, new_current, new_filled

    def loop_body(loop_state: BiLoopState) -> BiLoopState:
        """
        Main loop body for bidirectional A*.

        Strategy: Expand both directions in each iteration.
        """
        bi_result = loop_state.bi_result
        solve_config = loop_state.solve_config
        inverse_solveconfig = loop_state.inverse_solveconfig

        fwd_not_full = bi_result.forward.generated_size < bi_result.forward.capacity
        bwd_not_full = bi_result.backward.generated_size < bi_result.backward.capacity

        def _expand_forward(bi_result):
            return _expand_direction(
                bi_result,
                solve_config,
                inverse_solveconfig,
                loop_state.params_forward,
                loop_state.current_forward,
                loop_state.filled_forward,
                is_forward=True,
                use_heuristic=True,
            )

        def _expand_backward(bi_result):
            return _expand_direction(
                bi_result,
                solve_config,
                inverse_solveconfig,
                loop_state.params_backward,
                loop_state.current_backward,
                loop_state.filled_backward,
                is_forward=False,
                use_heuristic=use_backward_heuristic,
            )

        # Expand both directions
        bi_result, new_fwd_current, new_fwd_filled = jax.lax.cond(
            jnp.logical_and(loop_state.filled_forward.any(), fwd_not_full),
            _expand_forward,
            lambda br: (br, loop_state.current_forward, loop_state.filled_forward),
            bi_result,
        )

        bi_result, new_bwd_current, new_bwd_filled = jax.lax.cond(
            jnp.logical_and(loop_state.filled_backward.any(), bwd_not_full),
            _expand_backward,
            lambda br: (br, loop_state.current_backward, loop_state.filled_backward),
            bi_result,
        )

        return BiLoopState(
            bi_result=bi_result,
            solve_config=solve_config,
            inverse_solveconfig=inverse_solveconfig,
            params_forward=loop_state.params_forward,
            params_backward=loop_state.params_backward,
            current_forward=new_fwd_current,
            current_backward=new_bwd_current,
            filled_forward=new_fwd_filled,
            filled_backward=new_bwd_filled,
        )

    return init_loop_state, loop_condition, loop_body


def bi_astar_builder(
    puzzle: Puzzle,
    heuristic: Heuristic,
    batch_size: int = 1024,
    max_nodes: int = int(1e6),
    pop_ratio: float = jnp.inf,
    cost_weight: float = 1.0 - 1e-6,
    show_compile_time: bool = False,
    terminate_on_first_solution: bool = True,
    warmup_inputs: tuple[Puzzle.SolveConfig, Puzzle.State] | None = None,
):
    """
    Builds and returns a JAX-accelerated bidirectional A* search function.

    Bidirectional A* explores from both start and goal states simultaneously,
    meeting in the middle. This can significantly reduce the search space
    compared to unidirectional A*.

    Args:
        puzzle: Puzzle instance that defines the problem space and operations.
                Must support batched_get_inverse_neighbours for backward search.
        heuristic: Heuristic instance that provides state evaluation.
        batch_size: Number of states to process in parallel per direction.
        max_nodes: Maximum number of nodes to explore per direction.
        pop_ratio: Ratio controlling beam width.
        cost_weight: Weight for path cost in f = cost_weight * g + h.
        show_compile_time: If True, displays compilation time.

    Returns:
        A JIT-compiled function that performs bidirectional A* search.
    """
    statecls = puzzle.State
    action_size = puzzle.action_size
    denom = max(1, puzzle.action_size // 2)
    min_pop = max(1, MIN_BATCH_UNIT // denom)

    # Pre-build the search result OUTSIDE of JIT context
    bi_result_template = build_bi_search_result(
        statecls,
        batch_size,
        max_nodes,
        action_size,
        pop_ratio=pop_ratio,
        min_pop=min_pop,
        parant_with_costs=False,
    )

    use_backward_heuristic = not heuristic.is_fixed
    init_loop_state, loop_condition, loop_body = _bi_astar_loop_builder(
        puzzle,
        heuristic,
        bi_result_template,
        batch_size,
        cost_weight,
        use_backward_heuristic=use_backward_heuristic,
        terminate_on_first_solution=terminate_on_first_solution,
    )

    def bi_astar(
        solve_config: Puzzle.SolveConfig,
        start: Puzzle.State,
        **kwargs: Any,
    ) -> BiDirectionalSearchResult:
        """
        Perform bidirectional A* search.

        Args:
            solve_config: Configuration containing the goal state
            start: Initial state to search from

        Returns:
            BiDirectionalSearchResult containing both search trees and meeting point
        """
        # Prepare heuristic parameters for both directions
        heuristic_params_forward = heuristic.prepare_heuristic_parameters(solve_config, **kwargs)
        # Build a backward solve config that treats `start` as the target.
        # Prefer puzzle-level normalization via hindsight_transform.
        inverse_solveconfig = puzzle.hindsight_transform(solve_config, start)

        if use_backward_heuristic:
            heuristic_params_backward = heuristic.prepare_heuristic_parameters(
                inverse_solveconfig, **kwargs
            )
        else:
            heuristic_params_backward = heuristic_params_forward

        loop_state = init_loop_state(
            bi_result_template,
            solve_config,
            inverse_solveconfig,
            start,
            heuristic_params_forward,
            heuristic_params_backward,
        )
        loop_state = jax.lax.while_loop(loop_condition, loop_body, loop_state)

        bi_result = loop_state.bi_result

        # Mark as solved if meeting point was found
        bi_result.forward.solved = bi_result.meeting.found
        bi_result.forward.solved_idx = Current(
            hashidx=bi_result.meeting.fwd_hashidx,
            cost=bi_result.meeting.fwd_cost,
        )
        bi_result.backward.solved = bi_result.meeting.found
        bi_result.backward.solved_idx = Current(
            hashidx=bi_result.meeting.bwd_hashidx,
            cost=bi_result.meeting.bwd_cost,
        )

        return bi_result

    bi_astar_fn = jax.jit(bi_astar)
    if show_compile_time:
        print("Initializing JIT for bidirectional A*...")
        start_time = time.time()

    if warmup_inputs is None:
        empty_solve_config = puzzle.SolveConfig.default()
        empty_states = puzzle.State.default()
        # Pre-compile with empty data
        bi_astar_fn(empty_solve_config, empty_states)
    else:
        compile_with_example(bi_astar_fn, *warmup_inputs)

    if show_compile_time:
        end_time = time.time()
        print(f"Compile Time: {end_time - start_time:6.2f} seconds")
        print("JIT compiled\n")

    return bi_astar_fn
