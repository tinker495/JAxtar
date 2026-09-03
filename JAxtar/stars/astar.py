from typing import Any

import jax
import jax.numpy as jnp
import xtructure.numpy as xnp
from puxle import Puzzle

from helpers.jax_compile import compile_search_builder
from heuristic.heuristic_base import Heuristic
from JAxtar.annotate import ACTION_DTYPE, KEY_DTYPE, MIN_BATCH_SIZE
from JAxtar.search_build_spec import DEFAULT_SEARCH_BUILD_SPEC, SearchBuildSpec
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


def _row_heuristics(
    heuristic_fn,
    switcher,
    params,
    neighbours,
    vals: Current,
    new_states_mask,
    dist,
    batch_size: int,
    action_size: int,
):
    """Heuristic values for every action row, calling the network only where states are new.

    New states sit at the front of the partitioned batch, so the rows that need the
    network are a prefix: ``n_full`` all-new rows, then at most one mixed row when
    ``n_new % batch_size != 0``. Full rows call ``heuristic_fn`` directly inside a
    bounded while_loop (one host predicate readback per row instead of a lax.cond
    plus the batch switcher's lax.switch); the mixed row keeps the switcher (its
    ``inf`` padding past the evaluated slice is the existing behaviour); every other
    row reads the cached ``dist`` values. A row without new states must keep its
    cached values: the switcher has no zero branch (``MIN_BATCH_SIZE``), so an
    all-False mask would evaluate the first ``MIN_BATCH_SIZE`` entries and pad the
    rest with ``inf``, dropping improved old candidates from the queue.

    Returns the updated ``dist`` cache and ``heurs`` of shape ``(action_size, batch_size)``.
    """
    n_new = jnp.sum(new_states_mask, dtype=jnp.int32)
    n_full = n_new // batch_size
    heurs = dist[vals.hashidx.index]

    def _full_cond(carry):
        j, _, _ = carry
        return j < n_full

    def _full_body(carry):
        j, dist, heurs = carry
        row_heur = heuristic_fn(params, xnp.take(neighbours, j, axis=0)).astype(KEY_DTYPE)
        # cache the heuristic value; each newly inserted state owns one index
        dist = xnp.update_on_condition(
            dist, vals.hashidx.index[j], new_states_mask[j], row_heur, unique_indices=True
        )
        return j + 1, dist, heurs.at[j].set(row_heur)

    _, dist, heurs = jax.lax.while_loop(
        _full_cond, _full_body, (jnp.array(0, dtype=jnp.int32), dist, heurs)
    )

    has_mixed_row = jnp.logical_and(n_full < action_size, n_new % batch_size != 0)
    mixed = jnp.minimum(n_full, action_size - 1)
    mixed_mask = jnp.logical_and(new_states_mask[mixed], has_mixed_row)
    mixed_heur = switcher(params, xnp.take(neighbours, mixed, axis=0), mixed_mask).astype(KEY_DTYPE)
    dist = xnp.update_on_condition(
        dist, vals.hashidx.index[mixed], mixed_mask, mixed_heur, unique_indices=True
    )
    heurs = heurs.at[mixed].set(jnp.where(has_mixed_row, mixed_heur, heurs[mixed]))
    return dist, heurs


def _astar_loop_builder(
    puzzle: Puzzle,
    heuristic: Heuristic,
    batch_size: int = 1024,
    max_nodes: int = int(1e6),
    pop_ratio: float = jnp.inf,
    cost_weight: float = 1.0 - 1e-6,
    emit_workload_signature: bool = False,
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
            emit_workload_signature=emit_workload_signature,
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

        def _update_insert_stats(sr: SearchResult):
            cand_total_delta = jnp.sum(jnp.ones_like(flatten_filleds, dtype=jnp.int32)).astype(
                jnp.int32
            )
            return sr.replace(
                xtr_cand_total=sr.xtr_cand_total + cand_total_delta,
                xtr_cand_valid=sr.xtr_cand_valid + jnp.sum(flatten_filleds).astype(jnp.int32),
                xtr_cand_unique=sr.xtr_cand_unique
                + jnp.sum(cheapest_uniques_mask).astype(jnp.int32),
                xtr_ht_inserted=sr.xtr_ht_inserted
                + jnp.sum(flatten_new_states_mask).astype(jnp.int32),
            )

        search_result = jax.lax.cond(
            search_result.xtr_enabled,
            _update_insert_stats,
            lambda sr: sr,
            search_result,
        )

        # It must also be cheaper than any previously found path to this state.
        optimal_mask = jnp.less(flatten_nextcosts, search_result.get_cost(hash_idx))

        # Combine all conditions for the final decision.
        final_process_mask = jnp.logical_and(cheapest_uniques_mask, optimal_mask)

        def _update_accept_stats(sr: SearchResult):
            return sr.replace(
                xtr_accept=sr.xtr_accept + jnp.sum(final_process_mask).astype(jnp.int32)
            )

        search_result = jax.lax.cond(
            search_result.xtr_enabled,
            _update_accept_stats,
            lambda sr: sr,
            search_result,
        )

        # Update the cost (g-value) for the newly found optimal paths before they are
        # masked out. This ensures the cost table is always up-to-date.
        # cheapest_uniques_mask selects at most one entry per hash index, so the
        # masked scatter needs no first-true-wins resolution (unique_indices).
        search_result.cost = xnp.update_on_condition(
            search_result.cost,
            hash_idx.index,
            final_process_mask,
            flatten_nextcosts,  # Use costs before they are set to inf
            unique_indices=True,
        )
        search_result.parent = xnp.update_on_condition(
            search_result.parent,
            hash_idx.index,
            final_process_mask,
            flatten_parents,
            unique_indices=True,
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

        search_result.dist, neighbour_heurs = _row_heuristics(
            heuristic.batched_distance,
            variable_heuristic_batch_switcher,
            heuristic_parameters,
            neighbours,
            vals,
            new_states_mask,
            search_result.dist,
            sr_batch_size,
            action_size,
        )
        neighbour_keys = (cost_weight * vals.cost + neighbour_heurs).astype(KEY_DTYPE)
        search_result = insert_priority_queue_batches(
            search_result,
            neighbour_keys,
            vals,
            final_process_mask,
            prefix_rows=True,
            slim_carry=True,
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
    spec: SearchBuildSpec = DEFAULT_SEARCH_BUILD_SPEC,
):
    """
    Builds and returns a JAX-accelerated A* search function.

    Args:
        puzzle: Puzzle instance that defines the problem space and operations.
        heuristic: Heuristic instance that provides state evaluation.
        batch_size: Number of states to process in parallel (default: 1024).
        max_nodes: Maximum number of nodes to explore before terminating (default: 1e6).
        spec: Shared build-time tuning knobs for search construction.

    Returns:
        A function that performs A* search given a start state and solve configuration.
    """

    init_loop_state, loop_condition, loop_body = _astar_loop_builder(
        puzzle,
        heuristic,
        batch_size,
        max_nodes,
        spec.pop_ratio,
        spec.cost_weight,
        spec.emit_workload_signature,
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

    return compile_search_builder(astar, puzzle, spec.show_compile_time, spec.warmup_inputs)
