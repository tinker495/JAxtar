"""
Eager expansion core (A*).

An eager search generates and scores every successor of the popped batch before
queueing, so its priority queue holds `Current` states. Shared buffers and queue
helpers live in `common.py`.
"""

from functools import partial
from typing import Any

import jax
import jax.numpy as jnp
import xtructure.numpy as xnp
from puxle import Puzzle
from xtructure.core.dataclass import base_dataclass

from JAxtar.annotate import KEY_DTYPE
from JAxtar.stars.common import (
    POP_BATCH_FILLED_RATIO,
    SEARCH_RESULT_STATIC_FIELDS,
    Current,
    SearchResult,
    insert_start,
    record_pq_pop,
    search_continues,
    select_pop_batch,
)


@base_dataclass
class LoopState:
    """
    Loop state of the eager search (A*).
    Tracks hash indices, costs, and masks; states are fetched from the hash table on demand.
    """

    search_result: "EagerSearchResult"
    solve_config: Puzzle.SolveConfig
    params: Any
    current: Current
    filled: jax.Array


@base_dataclass(static_fields=SEARCH_RESULT_STATIC_FIELDS)
class EagerSearchResult(SearchResult[Current]):
    """Search result whose priority queue holds `Current` states (eager expansion)."""

    @staticmethod
    @partial(
        jax.jit,
        static_argnums=(0, 1, 2, 3, 4),
        static_argnames=("pop_ratio",),
    )
    def build(
        statecls: Puzzle.State,
        batch_size: int,
        max_nodes: int,
        action_size: int,
        pop_ratio: float = jnp.inf,
        min_pop: int = 1,
        seed=42,
        emit_workload_signature: bool = False,
    ) -> "EagerSearchResult":
        """Creates an eager search result; see `SearchResult._build` for the arguments."""
        return EagerSearchResult._build(
            Current,
            statecls,
            batch_size,
            max_nodes,
            action_size,
            pop_ratio,
            min_pop,
            seed,
            emit_workload_signature,
        )

    def pop_full(search_result) -> tuple["EagerSearchResult", Current, jax.Array]:
        """
        Removes and returns the minimum elements from the priority queue while maintaining
        the heap property. This function handles batched operations efficiently,
        respecting the pop_ratio to control search width without losing nodes.

        Args:
            search_result (SearchResult): The current search state

        Returns:
            tuple: Contains:
                - Updated SearchResult
                - A batch of the best values to be processed
                - A boolean mask indicating which entries in the batch are valid
        """

        # 1. Get an initial batch from the Priority Queue (PQ)
        search_result.priority_queue, min_key, min_val = search_result.priority_queue.delete_mins()
        delete_calls0 = jnp.array(1, dtype=jnp.int32)
        min_key = search_result.mask_unoptimal(min_key, min_val)
        popped0 = jnp.sum(jnp.isfinite(min_key)).astype(jnp.int32)
        batch_size = search_result.batch_size
        buffer = jnp.full((batch_size,), jnp.inf, dtype=KEY_DTYPE)
        buffer_val = Current.default((batch_size,))

        # 2. Loop to fill the batch if it's not full of valid nodes
        def _cond(state):
            search_result, key, _, _, _, _, _ = state
            pq_not_empty = search_result.priority_queue.size > 0

            # Optimization: Stop if batch is mostly full (e.g. 90%)
            # This prevents excessive sorting/merging for diminishing returns
            fill_ratio = jnp.mean(jnp.isfinite(key))
            not_full_enough = fill_ratio < POP_BATCH_FILLED_RATIO

            # For pop_ratio = inf, we don't check threshold and fill the batch completely.
            # For other cases, we can stop early, but the final filtering is done later.
            # This loop is mainly for ensuring we have enough candidates to select from.
            return jnp.logical_and(pq_not_empty, not_full_enough)

        def _body(state):
            search_result, key, val, _, _, delete_calls, popped_total = state
            # Pop new nodes from PQ
            (
                search_result.priority_queue,
                new_key,
                new_val,
            ) = search_result.priority_queue.delete_mins()
            new_key = search_result.mask_unoptimal(new_key, new_val)
            new_popped = jnp.sum(jnp.isfinite(new_key)).astype(jnp.int32)

            # Merge current batch with new nodes, splitting into main and overflow
            main_keys, main_vals, overflow_keys, overflow_vals = _unique_sort_merge_and_split(
                key, val, new_key, new_val, batch_size
            )
            return (
                search_result,
                main_keys,
                main_vals,
                overflow_keys,
                overflow_vals,
                delete_calls + jnp.array(1, dtype=jnp.int32),
                popped_total + new_popped,
            )

        # Run the loop until we have a full batch of the best available nodes
        (
            search_result,
            min_key,
            min_val,
            overflow_keys,
            overflow_vals,
            delete_calls,
            popped_total,
        ) = jax.lax.while_loop(
            _cond,
            _body,
            (search_result, min_key, min_val, buffer, buffer_val, delete_calls0, popped0),
        )

        # Put overflow nodes back into the PQ so they are never lost. Unconditional:
        # an all-inf block is a cheap no-op for BGPQ.insert, whereas the lax.cond cost
        # a host predicate readback plus a pass-through copy of the whole store per pop.
        search_result.priority_queue = search_result.priority_queue.insert(
            overflow_keys, overflow_vals
        )

        next_queue, final_process_mask = select_pop_batch(
            search_result.priority_queue,
            min_key,
            min_val,
            pop_ratio=search_result.pop_ratio,
            min_pop=search_result.min_pop,
            fixed_batch_pop=search_result.fixed_batch_pop,
        )

        search_result = record_pq_pop(search_result, final_process_mask, delete_calls, popped_total)

        search_result.priority_queue = next_queue

        # Same scalar for every selected index, so duplicates cannot disagree.
        search_result.pop_generation = xnp.update_on_condition(
            search_result.pop_generation,
            min_val.hashidx.index,
            final_process_mask,
            search_result.pop_count,
            unique_indices=True,
        )
        search_result.pop_count += 1

        return search_result, min_val, final_process_mask

    def mask_unoptimal(search_result, min_key: jax.Array, min_current: Current) -> jax.Array:
        """
        Mask the unoptimal states.
        """
        optimal = jnp.less_equal(min_current.cost, search_result.get_cost(min_current))
        return jnp.where(optimal, min_key, jnp.inf)


def unique_sort(key: jax.Array, val: Current, size: int | None = None) -> tuple[jax.Array, Current]:
    """
    Sorts the keys and corresponding values.

    Args:
        key (jax.Array): Array of keys.
        val (Current): Array of values.
        size (int | None): Optional static length for the sort index array.

    Returns:
        tuple:
            - sorted_key (jax.Array): Sorted array of keys.
            - sorted_val (Current): Values corresponding to the sorted keys.
    """
    n = key.shape[-1] if size is None else size
    filled = jnp.isfinite(key)
    mask = xnp.unique_mask(val.hashidx, val.cost, filled)
    key = jnp.where(mask, key, jnp.inf)
    sorted_key, sorted_idx = jax.lax.sort_key_val(key, jnp.arange(n))
    sorted_val = val[sorted_idx]
    return sorted_key, sorted_val


# Helper to merge, sort, and split two batches of nodes
def _unique_sort_merge_and_split(k1, v1, k2, v2, batch_size: int):
    merged_key = jnp.concatenate([k1, k2])
    merged_val = xnp.concatenate([v1, v2])
    merged_size = batch_size * 2

    # Sort and remove duplicates from the combined batch
    sorted_key, sorted_val = unique_sort(merged_key, merged_val, merged_size)

    # Split into a main batch and an overflow batch
    main_keys = sorted_key[:batch_size]
    main_vals = sorted_val[:batch_size]
    overflow_keys = sorted_key[batch_size:]
    overflow_vals = sorted_val[batch_size:]
    return main_keys, main_vals, overflow_keys, overflow_vals


def init_base_loop_state(
    puzzle: Puzzle,
    search_result: EagerSearchResult,
    solve_config: Puzzle.SolveConfig,
    start: Puzzle.State,
    params: Any,
    sr_batch_size: int,
):
    hash_idx = insert_start(search_result, start)
    hash_idxs = xnp.pad(hash_idx, (0, sr_batch_size - 1))
    costs = jnp.full((sr_batch_size,), jnp.inf, dtype=KEY_DTYPE).at[0].set(0)
    filled = jnp.zeros(sr_batch_size, dtype=jnp.bool_).at[0].set(True)

    return LoopState(
        search_result=search_result,
        solve_config=solve_config,
        params=params,
        current=Current(hashidx=hash_idxs, cost=costs),
        filled=filled,
    )


def base_loop_condition(puzzle: Puzzle, loop_state: LoopState):
    search_result = loop_state.search_result
    states = search_result.get_state(loop_state.current)
    return search_continues(
        puzzle, search_result, loop_state.solve_config, states, loop_state.filled
    )
