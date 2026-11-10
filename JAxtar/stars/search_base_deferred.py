"""
Deferred expansion core (A*-D, Q*).

A deferred search queues each (parent, action) edge with a score derived from the
parent and generates the successor only when the edge is popped, so its priority
queue holds `ParentWithCosts` entries. Shared buffers and queue helpers live in
`common.py`.
"""

from dataclasses import replace
from functools import partial
from typing import Annotated, Any, Callable

import jax
import jax.numpy as jnp
import xtructure.numpy as xnp
from puxle import Puzzle
from xtructure import FieldDescriptor, Xtructurable
from xtructure.core.dataclass import base_dataclass

from JAxtar.annotate import ACTION_DTYPE, KEY_DTYPE
from JAxtar.stars.common import (
    POP_BATCH_FILLED_RATIO,
    SEARCH_RESULT_STATIC_FIELDS,
    Current,
    Parent,
    SearchResult,
    insert_priority_queue_batches,
    insert_start,
    record_pq_pop,
    search_continues,
    select_pop_batch,
)


class ParentWithCosts(Xtructurable):
    """
    Priority-queue value of the deferred search: a (parent, action) edge with the
    parent's cost and the score inputs carried until the edge is popped.
    """

    parent: Annotated[Parent, FieldDescriptor.scalar(dtype=Parent)]
    cost: Annotated[jax.Array, FieldDescriptor.scalar(dtype=KEY_DTYPE)]
    dist: Annotated[jax.Array, FieldDescriptor.scalar(dtype=KEY_DTYPE)]


@base_dataclass
class LoopStateWithStates:
    """
    Loop state of the deferred search (A*-D, Q*); carries the already-expanded current states.

    This avoids an extra `HashTable.__getitem__` / gather of the same states in both
    the loop condition and body; the states were already materialized during
    `pop_full_with_actions`.
    """

    search_result: "DeferredSearchResult"
    solve_config: Puzzle.SolveConfig
    params: Any
    current: Current
    states: Puzzle.State
    filled: jax.Array


@base_dataclass(static_fields=SEARCH_RESULT_STATIC_FIELDS + ("estimated_successors",))
class DeferredSearchResult(SearchResult[ParentWithCosts]):
    """
    Search result whose priority queue holds `ParentWithCosts` edges (deferred expansion).

    Attributes:
        estimated_successors (int): Successors one expansion contributes, i.e.
                `action_size` minus the parent for reversible puzzles. Only used to
                scale the generated-state metric.
    """

    estimated_successors: int  # action_size - 1 if reversible

    @staticmethod
    @partial(
        jax.jit,
        static_argnums=(0, 1, 2, 3, 4),
        static_argnames=("pop_ratio", "is_reversible"),
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
        is_reversible: bool = False,
    ) -> "DeferredSearchResult":
        """
        Creates a deferred search result; see `SearchResult._shared_buffers` for the
        shared arguments.

        Args:
            is_reversible (bool): Whether one of the `action_size` children of an
                expansion is its own parent, which the generated-state metric discounts.
        """
        return DeferredSearchResult(
            **SearchResult._shared_buffers(
                ParentWithCosts,
                statecls,
                batch_size,
                max_nodes,
                action_size,
                pop_ratio,
                min_pop,
                seed,
                emit_workload_signature,
            ),
            estimated_successors=action_size - 1 if is_reversible else action_size,
        )

    @property
    def generated_size(self) -> int:
        """Number of states the search generated.

        Deferred expansion only inserts the nodes it pops and processes, so hash usage
        counts *expansions*; scale it by the estimated successor count to keep the
        metric comparable with eager expansion. Use `hashtable.size` directly
        wherever hash occupancy itself is meant.
        """
        return self.hashtable.size * self.estimated_successors

    def pop_full_with_actions(
        search_result,
        puzzle: Puzzle,
        solve_config: Puzzle.SolveConfig,
        use_heuristic: bool = False,
        is_backward: bool = False,
    ) -> tuple["DeferredSearchResult", Current, Puzzle.State, jax.Array]:
        """
        Removes and returns the minimum elements from the priority queue while maintaining
        the heap property. This function handles batched operations efficiently,
        respecting the pop_ratio to control search width without losing nodes.

        Each popped (S, A) edge is applied to generate S' inside the refill loop, and
        S' is deduplicated there, so the returned batch contains only unique S'
        states; this prevents batch starvation in highly reversible environments.

        Args:
            search_result (DeferredSearchResult): The current search state
            puzzle (Puzzle): The puzzle instance
            solve_config (Puzzle.SolveConfig): Configuration for the puzzle

        Returns:
            tuple: Contains:
                - Updated DeferredSearchResult
                - A batch of the best values to be processed (Current)
                - The materialized states of that batch
                - A boolean mask indicating which entries in the batch are valid
        """

        def _expand_and_filter(search_result, key, val):
            """Helper to expand nodes, lookup in hashtable, and filter unoptimal ones."""
            filled = jnp.isfinite(key)
            parent_states = search_result.get_state(val.parent)
            parent_actions = val.parent.action  # [batch_size]
            parent_costs = search_result.get_cost(val.parent)

            if is_backward:
                # In bidirectional deferred search, backward expansion stores action indices
                # that refer to the i-th inverse neighbour. Child generation must therefore
                # use inverse neighbours and gather by action index.
                if puzzle.is_reversible:
                    inv_map = puzzle.inverse_action_map  # [action_size]
                    inv_actions = inv_map[parent_actions.astype(jnp.int32)]
                    # Use batched_get_actions for efficient O(batch) expansion
                    current_states, ncosts = puzzle.batched_get_actions(
                        solve_config, parent_states, inv_actions, filled
                    )
                else:
                    all_inv_neighbors, all_inv_costs = puzzle.batched_get_inverse_neighbours(
                        solve_config, parent_states, filled
                    )  # [action, batch, ...], [action, batch]

                    parent_actions_i32 = parent_actions.astype(jnp.int32)
                    batch_indices = jnp.arange(parent_actions_i32.shape[0], dtype=jnp.int32)
                    current_states = all_inv_neighbors[parent_actions_i32, batch_indices]
                    ncosts = all_inv_costs[parent_actions_i32, batch_indices]
            else:
                current_states, ncosts = puzzle.batched_get_actions(
                    solve_config, parent_states, parent_actions, filled
                )  # [batch_size]

            current_costs = (parent_costs + ncosts).astype(KEY_DTYPE)
            current_dists = (val.dist if use_heuristic else (val.dist - ncosts)).astype(KEY_DTYPE)

            # Optimization: deduplicate before lookup to avoid redundant hash lookups.
            unique_mask = xnp.unique_mask(current_states, current_costs, filled)
            # Capture the hash-pass probe here so the later parallel_insert of these
            # same states can reuse it instead of hashing them a second time. The probe
            # is per-state and independent of the mask, so it stays valid as the states
            # are permuted/merged before insertion (we carry it through identically).
            (
                current_hash_idxs,
                found,
                current_probe,
            ) = search_result.hashtable.lookup_parallel_with_probe(current_states, unique_mask)

            def _update_lookup(sr: "DeferredSearchResult"):
                return replace(
                    sr,
                    xtr_ht_lookup=sr.xtr_ht_lookup + jnp.sum(unique_mask).astype(jnp.int32),
                    xtr_ht_found=sr.xtr_ht_found
                    + jnp.sum(jnp.logical_and(found, unique_mask)).astype(jnp.int32),
                )

            search_result = jax.lax.cond(
                search_result.xtr_enabled,
                _update_lookup,
                lambda sr: sr,
                search_result,
            )

            old_costs = search_result.get_cost(current_hash_idxs)
            # Only consider nodes that are either:
            # 1. Not found in the hash table (new nodes), or
            # 2. Found but have better cost than existing
            better_cost_mask = jnp.less(current_costs, old_costs)
            optimal_mask = unique_mask & (jnp.logical_or(~found, better_cost_mask))

            # Mask unoptimal keys with infinity
            filtered_key = jnp.where(optimal_mask, key, jnp.inf)

            return (
                search_result,
                current_states,
                current_costs,
                current_dists,
                filtered_key,
                current_probe,
            )

        # 1. Get an initial batch from the Priority Queue (PQ)
        search_result.priority_queue, min_key, min_val = search_result.priority_queue.delete_mins()
        delete_calls0 = jnp.array(1, dtype=jnp.int32)
        batch_size = search_result.batch_size
        stack_size = batch_size * 2

        # Initial expansion and filtering
        (
            search_result,
            current_states,
            current_costs,
            current_dists,
            min_key,
            current_probe,
        ) = _expand_and_filter(search_result, min_key, min_val)
        popped0 = jnp.sum(jnp.isfinite(min_key)).astype(jnp.int32)

        # Apply unique mask to initial batch to maintain invariant
        filled = jnp.isfinite(min_key)
        unique_mask = xnp.unique_mask(current_states, current_costs, filled)
        min_key = jnp.where(unique_mask, min_key, jnp.inf)

        # Initialize buffer for overflow (inf)
        buffer_key = jnp.full((batch_size,), jnp.inf, dtype=KEY_DTYPE)
        buffer_val = ParentWithCosts.default((batch_size,))
        # Carry the per-state hash probe alongside states through the merge/sort loop
        # so the final batch's probe stays aligned with final_states for reuse at insert.
        buffer_probe = jax.tree.map(jnp.zeros_like, current_probe)

        # 2. Loop to fill the batch if it's not full of valid nodes
        def _cond(state):
            search_result, _, _, _, key, _, _, _, _, _, _, _ = state
            pq_not_empty = search_result.priority_queue.size > 0

            # Optimization: Stop if batch is mostly full (e.g. 90%)
            fill_ratio = jnp.mean(jnp.isfinite(key))
            not_full_enough = fill_ratio < POP_BATCH_FILLED_RATIO

            return jnp.logical_and(pq_not_empty, not_full_enough)

        def _body(state):
            (
                search_result,
                current_states,
                costs,
                dists,
                key,
                val,
                _,
                _,
                delete_calls,
                popped_total,
                probe,
                _buffer_probe,
            ) = state

            # Pop new nodes from PQ
            (
                search_result.priority_queue,
                new_key,
                new_val,
            ) = search_result.priority_queue.delete_mins()

            # Expand and filter new nodes
            (
                search_result,
                new_states,
                new_costs,
                new_dists,
                new_key,
                new_probe,
            ) = _expand_and_filter(search_result, new_key, new_val)
            new_popped = jnp.sum(jnp.isfinite(new_key)).astype(jnp.int32)

            # Merge with existing batch
            stack_states = xnp.concatenate((current_states, new_states))
            stack_costs = jnp.concatenate((costs, new_costs))
            stack_dists = jnp.concatenate((dists, new_dists))
            stack_key = jnp.concatenate((key, new_key))
            stack_val = xnp.concatenate((val, new_val))
            # Concatenate the hash probes in lockstep with the states.
            stack_probe = jax.tree.map(lambda a, b: jnp.concatenate([a, b]), probe, new_probe)

            # Deduplicate across the entire stack (old + new)
            stack_filled = jnp.isfinite(stack_key)
            unique_mask = xnp.unique_mask(stack_states, stack_costs, stack_filled)
            stack_key = jnp.where(unique_mask, stack_key, jnp.inf)

            # Sort to keep best candidates
            sorted_key, sorted_idx = jax.lax.sort_key_val(stack_key, jnp.arange(stack_size))
            sorted_val = stack_val[sorted_idx]
            sorted_states = stack_states[sorted_idx]
            sorted_costs = stack_costs[sorted_idx]
            sorted_dists = stack_dists[sorted_idx]
            # Permute the probe by the same order so probe[i] stays the hash of states[i].
            sorted_probe = jax.tree.map(lambda a: a[sorted_idx], stack_probe)

            # Split back into current batch and overflow buffer
            return (
                search_result,
                sorted_states[:batch_size],
                sorted_costs[:batch_size],
                sorted_dists[:batch_size],
                sorted_key[:batch_size],
                sorted_val[:batch_size],
                sorted_key[batch_size:],  # New buffer (overflow)
                sorted_val[batch_size:],  # New buffer val
                delete_calls + jnp.array(1, dtype=jnp.int32),
                popped_total + new_popped,
                jax.tree.map(lambda a: a[:batch_size], sorted_probe),
                jax.tree.map(lambda a: a[batch_size:], sorted_probe),
            )

        # Run the loop until we have a full batch of the best available nodes
        (
            search_result,
            final_states,
            final_costs,
            final_dists,
            final_key,
            final_val,
            overflow_keys,
            overflow_vals,
            delete_calls,
            popped_total,
            final_probe,
            _final_buffer_probe,
        ) = jax.lax.while_loop(
            _cond,
            _body,
            (
                search_result,
                current_states,
                current_costs,
                current_dists,
                min_key,
                min_val,
                buffer_key,
                buffer_val,
                delete_calls0,
                popped0,
                current_probe,
                buffer_probe,
            ),
        )

        final_parents = final_val.parent

        # Put overflow nodes back into the PQ so they are never lost. Unconditional:
        # an all-inf block is a cheap no-op for BGPQ.insert, whereas the lax.cond cost
        # a host predicate readback plus a pass-through copy of the whole store per pop.
        search_result.priority_queue = search_result.priority_queue.insert(
            overflow_keys, overflow_vals
        )

        next_queue, final_process_mask = select_pop_batch(
            search_result.priority_queue,
            final_key,
            final_val,
            pop_ratio=search_result.pop_ratio,
            min_pop=search_result.min_pop,
            fixed_batch_pop=search_result.fixed_batch_pop,
        )

        search_result = record_pq_pop(search_result, final_process_mask, delete_calls, popped_total)

        final_costs = jnp.where(final_process_mask, final_costs, jnp.inf)
        search_result.priority_queue = next_queue

        # The merge loop already deduplicated these states. Reuse its probe and
        # skip both redundant hashing and intra-batch deduplication at insertion.
        (
            search_result.hashtable,
            inserted_mask,
            _,
            hash_idx,
        ) = search_result.hashtable.parallel_insert(
            final_states,
            final_process_mask,
            probe=final_probe,
            assume_unique=True,
        )

        def _update_inserted(sr: "DeferredSearchResult"):
            return replace(
                sr, xtr_ht_inserted=sr.xtr_ht_inserted + jnp.sum(inserted_mask).astype(jnp.int32)
            )

        search_result = jax.lax.cond(
            search_result.xtr_enabled,
            _update_inserted,
            lambda sr: sr,
            search_result,
        )

        final_currents = Current(hashidx=hash_idx, cost=final_costs)

        # The merge loop deduplicated final_states (assume_unique above), so each
        # selected hash index appears once: masked scatter without first-true-wins.
        search_result.cost = xnp.update_on_condition(
            search_result.cost,
            hash_idx.index,
            final_process_mask,
            final_costs,
            unique_indices=True,
        )
        search_result.dist = xnp.update_on_condition(
            search_result.dist,
            hash_idx.index,
            final_process_mask,
            final_dists,
            unique_indices=True,
        )
        search_result.parent = xnp.update_on_condition(
            search_result.parent,
            hash_idx.index,
            final_process_mask,
            final_parents,
            unique_indices=True,
        )
        search_result.pop_generation = xnp.update_on_condition(
            search_result.pop_generation,
            hash_idx.index,
            final_process_mask,
            search_result.pop_count,
            unique_indices=True,
        )
        search_result.pop_count += 1

        return search_result, final_currents, final_states, final_process_mask


def init_base_loop_state(
    puzzle: Puzzle,
    search_result: DeferredSearchResult,
    solve_config: Puzzle.SolveConfig,
    start: Puzzle.State,
    params: Any,
    sr_batch_size: int,
):
    hash_idx = insert_start(search_result, start)
    costs = jnp.zeros((sr_batch_size,), dtype=KEY_DTYPE)
    states = xnp.pad(start, (0, sr_batch_size - 1))
    hash_idxs = xnp.pad(hash_idx, (0, sr_batch_size - 1))
    filled = jnp.zeros(sr_batch_size, dtype=jnp.bool_).at[0].set(True)

    return LoopStateWithStates(
        search_result=search_result,
        solve_config=solve_config,
        params=params,
        current=Current(hashidx=hash_idxs, cost=costs),
        states=states,
        filled=filled,
    )


def base_loop_condition(puzzle: Puzzle, loop_state: LoopStateWithStates):
    return search_continues(
        puzzle,
        loop_state.search_result,
        loop_state.solve_config,
        loop_state.states,
        loop_state.filled,
    )


def build_deferred_loop_body(
    puzzle: Puzzle,
    look_ahead_pruning: bool,
    cost_weight: float,
    eval_fn: Callable,  # Returns neighbour_keys, dists, optimal_mask (or handles its own pruning)
    use_heuristic: bool = False,
):
    def loop_body(loop_state: LoopStateWithStates):
        search_result = loop_state.search_result
        solve_config = loop_state.solve_config
        params = loop_state.params
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

        # Inject standard computations, delegate to eval_fn for Q-value / Heuristic specific logic
        neighbour_keys, dists, optimal_mask = eval_fn(
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
        )

        flattened_vals = ParentWithCosts(
            parent=Parent(hashidx=idx_tiles.flatten(), action=action.flatten()),
            cost=costs.flatten(),
            dist=dists.flatten(),
        )

        flattened_neighbour_keys = jnp.where(
            optimal_mask.flatten(), neighbour_keys.flatten(), jnp.inf
        )

        # Sort to keep best candidates
        sorted_key, sorted_idx = jax.lax.sort_key_val(
            flattened_neighbour_keys, jnp.arange(flat_size)
        )
        sorted_vals = flattened_vals[sorted_idx]
        optimal_mask_sorted = optimal_mask.flatten()[sorted_idx]

        neighbour_keys_out = sorted_key.reshape(action_size, sr_batch_size)
        vals_out = sorted_vals.reshape((action_size, sr_batch_size))
        optimal_mask_out = optimal_mask_sorted.reshape(action_size, sr_batch_size)

        # The global sort put every finite key first, so the rows holding
        # candidates are a prefix: one multi-row insert of those rows.
        search_result = insert_priority_queue_batches(
            search_result,
            neighbour_keys_out,
            vals_out,
            optimal_mask_out,
            presorted=True,
            prefix_rows=True,
        )
        search_result, min_val, next_states, next_filled = search_result.pop_full_with_actions(
            puzzle=puzzle, solve_config=solve_config, use_heuristic=use_heuristic
        )

        return LoopStateWithStates(
            search_result=search_result,
            solve_config=solve_config,
            params=params,
            current=min_val,
            states=next_states,
            filled=next_filled,
        )

    return loop_body
