"""
JAxtar Search Base Module
This module implements the core search functionality for A*/Q* algorithms using JAX.
The implementation is designed to be fully parallelizable and GPU-compatible.
Key features:
- Pure JAX implementation for ML research
- Batched operations for GPU optimization
- Generic puzzle-agnostic implementation
"""

from functools import partial
from typing import Any

import chex
import jax
import jax.numpy as jnp
import xtructure.numpy as xnp
from puxle import Puzzle
from xtructure import (
    BGPQ,
    FieldDescriptor,
    HashIdx,
    HashTable,
    Xtructurable,
    base_dataclass,
    xtructure_dataclass,
)

from JAxtar.annotate import (
    ACTION_DTYPE,
    CUCKOO_TABLE_N,
    HASH_SIZE_MULTIPLIER,
    KEY_DTYPE,
)

POP_BATCH_FILLED_RATIO = 0.99  # ratio of batch to be filled before popping


def print_states(states: Xtructurable, costs: chex.Array, dists: chex.Array, key: chex.Array):
    print(states)
    print(f"costs: {costs}")
    print(f"dists: {dists}")
    print(f"key: {key}")


def print_states_w_actions(states: Xtructurable, costs: chex.Array, actions: chex.Array):
    print(states)
    print(f"costs: {costs}")
    print(f"actions: {actions}")


@partial(jax.jit, static_argnames=("max_steps",))
def _reconstruct_path_arrays(
    parent_indices: chex.Array,
    costs: chex.Array,
    dists: chex.Array,
    solved_index: chex.Array,
    max_steps: int,
) -> tuple[chex.Array, chex.Array, chex.Array, chex.Array, chex.Array, chex.Array]:
    path_idx = jnp.full((max_steps,), -1, dtype=jnp.int32)
    path_costs = jnp.full((max_steps,), jnp.nan, dtype=costs.dtype)
    path_dists = jnp.full((max_steps,), jnp.nan, dtype=dists.dtype)

    def _cond(carry):
        idx, step, *_ = carry
        return jnp.logical_and(idx >= 0, step < max_steps)

    def _body(carry):
        idx, step, path_idx, path_costs, path_dists = carry
        path_idx = path_idx.at[step].set(idx)
        path_costs = path_costs.at[step].set(costs[idx])
        path_dists = path_dists.at[step].set(dists[idx])
        next_idx = parent_indices[idx]
        return next_idx, step + 1, path_idx, path_costs, path_dists

    init = (
        solved_index.astype(jnp.int32),
        jnp.array(0, dtype=jnp.int32),
        path_idx,
        path_costs,
        path_dists,
    )
    _, length, path_idx, path_costs, path_dists = jax.lax.while_loop(_cond, _body, init)

    def _detect_loop():
        used = path_idx[:length]
        sorted_used = jnp.sort(used)
        return jnp.any(sorted_used[1:] == sorted_used[:-1])

    def _detect_corruption():
        used_costs = path_costs[:length]
        reversed_costs = jnp.flip(used_costs, axis=0)
        return jnp.any(reversed_costs[1:] < reversed_costs[:-1])

    loop_detected = jax.lax.cond(length > 1, _detect_loop, lambda: jnp.array(False))
    corruption_detected = jax.lax.cond(length > 1, _detect_corruption, lambda: jnp.array(False))
    return path_idx, path_costs, path_dists, length, loop_detected, corruption_detected


@xtructure_dataclass
class Parent:

    hashidx: FieldDescriptor.scalar(dtype=HashIdx)
    action: FieldDescriptor.scalar(dtype=ACTION_DTYPE)


@xtructure_dataclass
class Current:

    hashidx: FieldDescriptor.scalar(dtype=HashIdx)
    cost: FieldDescriptor.scalar(dtype=KEY_DTYPE)


@xtructure_dataclass
class Parant_with_Costs:
    """
    A dataclass representing a hash table heap value for the priority queue.
    This class maintains the mapping between states in the hash table and their positions.
    """

    parent: FieldDescriptor.scalar(dtype=Parent)
    cost: FieldDescriptor.scalar(dtype=KEY_DTYPE)
    dist: FieldDescriptor.scalar(dtype=KEY_DTYPE)


@base_dataclass(static_fields=("params"))
class LoopState:
    """
    Unified loop state for search loops (A*, Q*, A*_d).
    Tracks hash indices, costs, and masks; states can be fetched on demand.
    """

    search_result: "SearchResult"
    solve_config: Puzzle.SolveConfig
    params: Any
    current: Current
    filled: chex.Array


@base_dataclass(static_fields=("params"))
class LoopStateWithStates:
    """
    Loop state that carries the already-expanded current states.

    This avoids an extra `HashTable.__getitem__` / gather of the same states in both
    the loop condition and body (notably in deferred variants where the states were
    already materialized during `pop_full_with_actions`).
    """

    search_result: "SearchResult"
    solve_config: Puzzle.SolveConfig
    params: Any
    current: Current
    states: Puzzle.State
    filled: chex.Array


@base_dataclass(static_fields=("action_size",))
class SearchResult:
    """
    A dataclass containing the data structures used in the A*/Q* search algorithms.
    This class maintains the state of the search process, including open and closed sets,
    priority queue, and path tracking information.

    Implementation Notes:
    - Uses a HashTable for efficient state storage and lookup
    - Maintains a priority queue (BGPQ) for state expansion ordering
    - Tracks costs and parent relationships for path reconstruction
    - Optimized for GPU execution with batched operations

    Attributes:
        hashtable (HashTable): Stores all encountered states for efficient lookup
        priority_queue (BGPQ): Priority queue for ordering state expansions
        action_size (int): Number of actions available from each state
        pop_ratio (float): The final pop ratio for the search, controlling the
                beam width. It is used to calculate a multiplier `M = max(1.0 + pop_ratio, 1.01)`.
                The search beam will include all nodes with a cost up to `min_cost * M`.
        min_pop (int): Minimum number of states to pop from the priority queue
        cost (chex.Array): Cost array tracking path costs to each state (g value)
        dist (chex.Array): Distance array storing calculated heuristic or Q values
        parent (Parent): Array storing parent state indices for path reconstruction
        solved (chex.Array): Boolean flag indicating if a solution has been found
        solved_idx (Current): Index of the solved state in the hash table
    """

    hashtable: HashTable  # hash table
    priority_queue: BGPQ  # priority queue
    action_size: int  # number of actions available from each state
    pop_ratio: float  # ratio of states to pop from the priority queue
    min_pop: int  # minimum number of states to pop from the priority queue
    cost: chex.Array  # cost array - g value
    dist: chex.Array  # distance array - calculated heuristic or Q value
    parent: Xtructurable | Parent  # parent array
    solved: chex.Array  # solved array
    solved_idx: Xtructurable | Current  # solved index
    pop_generation: chex.Array  # records which pop generation a node was expanded in
    pop_count: chex.Array  # counter for pop_full calls

    @staticmethod
    @partial(jax.jit, static_argnums=(0, 1, 2, 3), static_argnames=("parant_with_costs",))
    def build(
        statecls: Puzzle.State,
        batch_size: int,
        max_nodes: int,
        action_size: int,
        pop_ratio: float = jnp.inf,
        min_pop: int = 1,
        seed=42,
        parant_with_costs: bool = False,
    ):
        """
        Creates a new instance of SearchResult with initialized data structures.

        Args:
            statecls (Puzzle.State): The state class for the puzzle being solved
            batch_size (int): Size of batches for parallel processing
            max_nodes (int): Maximum number of nodes to store
            action_size (int): Number of actions available from each state
            pop_ratio (float): Controls the search beam width. It is used to calculate
                a multiplier `M = max(1.0 + pop_ratio, 1.01)`. The search beam will
                include all nodes with a cost up to `min_cost * M`.
                - `pop_ratio = 0`: Results in a narrow beam (`M=1.01`).
                - `pop_ratio > 0`: Creates a wider beam. A larger `pop_ratio`
                                 results in a wider beam.
                - `pop_ratio = inf`: Effectively becomes batched A*, processing all
                                 available nodes in the batch.
            min_pop (int): Minimum number of nodes to pop from the priority queue.
            seed (int): Random seed for hash function initialization

        Returns:
            SearchResult: A new instance with initialized data structures
        """
        # Initialize the hash table for state storage
        hashtable: HashTable = HashTable.build(
            statecls, seed, max_nodes, CUCKOO_TABLE_N, HASH_SIZE_MULTIPLIER
        )

        if parant_with_costs:
            # Initialize priority queue for state expansion
            priority_queue = BGPQ.build(max_nodes, batch_size, Parant_with_Costs, KEY_DTYPE)
        else:
            priority_queue = BGPQ.build(max_nodes, batch_size, Current, KEY_DTYPE)

        # Initialize arrays for tracking costs and state relationships
        # +1 for -1 index as a dummy node
        cost = jnp.full(hashtable.table.shape.batch, jnp.inf, dtype=KEY_DTYPE)
        dist = jnp.full(hashtable.table.shape.batch, jnp.inf, dtype=KEY_DTYPE)
        parent = Parent.default(hashtable.table.shape.batch)
        solved = jnp.array(False)
        solved_idx = Current.default((1,))
        pop_generation = jnp.full(hashtable.table.shape.batch, -1, dtype=jnp.int32)
        pop_count = jnp.array(0, dtype=jnp.int32)

        return SearchResult(
            hashtable=hashtable,
            priority_queue=priority_queue,
            action_size=action_size,
            pop_ratio=jnp.maximum(1.0 + pop_ratio, 1.01),
            min_pop=min_pop,
            cost=cost,
            dist=dist,
            parent=parent,
            solved=solved,
            solved_idx=solved_idx,
            pop_generation=pop_generation,
            pop_count=pop_count,
        )

    @property
    def capacity(self) -> int:
        """Maximum number of states that can be stored."""
        return self.hashtable.capacity

    @property
    def batch_size(self) -> int:
        """Batch size of the search."""
        return self.priority_queue.batch_size

    @property
    def generated_size(self) -> int:
        """Current number of states stored in the founded set.
        This is the number of states in the founded set."""
        return self.hashtable.size

    def pop_full(search_result, **kwargs) -> tuple["SearchResult", Current, chex.Array]:
        if isinstance(search_result.priority_queue.val_store, Current):
            return search_result._pop_full_with_current(**kwargs)
        else:
            return search_result._pop_full_with_parent_with_costs(**kwargs)

    def pop_full_with_actions(
        search_result,
        puzzle: Puzzle,
        solve_config: Puzzle.SolveConfig,
        use_heuristic: bool = False,
        **kwargs,
    ) -> tuple["SearchResult", Current, Puzzle.State, chex.Array]:
        """
        Pop a full batch and also return the corresponding materialized states.

        - For standard A* (`Current` PQ values), states are gathered from the hash table.
        - For deferred variants (`Parant_with_Costs` PQ values), states are already
          computed during the eager expansion inside the pop routine, so returning
          them avoids re-gathering the same batch again in the caller.
        """

        if isinstance(search_result.priority_queue.val_store, Current):
            search_result, current, filled = search_result._pop_full_with_current(**kwargs)
            states = search_result.get_state(current)
            return search_result, current, states, filled

        return search_result._pop_full_with_parent_with_costs(
            puzzle=puzzle,
            solve_config=solve_config,
            use_heuristic=use_heuristic,
            return_states=True,
            **kwargs,
        )

    def _pop_full_with_current(
        search_result, **kwargs
    ) -> tuple["SearchResult", Current, chex.Array]:
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
        min_key = search_result.mask_unoptimal(min_key, min_val)
        batch_size = search_result.batch_size
        buffer = jnp.full((batch_size,), jnp.inf, dtype=KEY_DTYPE)
        buffer_val = Current.default((batch_size,))

        # 2. Loop to fill the batch if it's not full of valid nodes
        def _cond(state):
            search_result, key, _, _, _ = state
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
            search_result, key, val, _, _ = state
            # Pop new nodes from PQ
            (
                search_result.priority_queue,
                new_key,
                new_val,
            ) = search_result.priority_queue.delete_mins()
            new_key = search_result.mask_unoptimal(new_key, new_val)

            # Merge current batch with new nodes, splitting into main and overflow
            main_keys, main_vals, overflow_keys, overflow_vals = _unique_sort_merge_and_split(
                key, val, new_key, new_val, batch_size
            )
            return search_result, main_keys, main_vals, overflow_keys, overflow_vals

        # Run the loop until we have a full batch of the best available nodes
        search_result, min_key, min_val, overflow_keys, overflow_vals = jax.lax.while_loop(
            _cond, _body, (search_result, min_key, min_val, buffer, buffer_val)
        )

        # Put overflow nodes back into the PQ so they are never lost
        search_result.priority_queue = jax.lax.cond(
            jnp.any(jnp.isfinite(overflow_keys)),
            lambda: search_result.priority_queue.insert(overflow_keys, overflow_vals),
            lambda: search_result.priority_queue,
        )

        # 3. Apply pop_ratio to the full batch
        filled = jnp.isfinite(min_key)
        # Add a small epsilon for floating point comparisons
        threshold = min_key[0] * search_result.pop_ratio + 1e-6

        # Identify nodes to process now vs. nodes to return to PQ
        process_mask = jnp.less_equal(min_key, threshold)
        base_process_mask = jnp.logical_and(filled, process_mask)

        # Enforce min_pop: ensure that we pop at least min_pop nodes if they are available.
        min_pop_mask = jnp.logical_and(jnp.cumsum(filled) <= search_result.min_pop, filled)

        # The final mask includes nodes within the threshold OR nodes to meet min_pop.
        final_process_mask = jnp.logical_or(base_process_mask, min_pop_mask)

        # Separate the nodes to be returned and re-insert them into the PQ
        return_keys = jnp.where(final_process_mask, jnp.inf, min_key)
        search_result.priority_queue = search_result.priority_queue.insert(return_keys, min_val)

        search_result.pop_generation = xnp.update_on_condition(
            search_result.pop_generation,
            min_val.hashidx.index,
            final_process_mask,
            search_result.pop_count,
        )
        search_result.pop_count += 1

        return search_result, min_val, final_process_mask

    def _pop_full_with_parent_with_costs(
        search_result,
        puzzle: Puzzle,
        solve_config: Puzzle.SolveConfig,
        use_heuristic: bool = False,
        return_states: bool = False,
        **kwargs,
    ) -> tuple["SearchResult", Current, chex.Array] | tuple[
        "SearchResult", Current, Puzzle.State, chex.Array
    ]:
        """
        Removes and returns the minimum elements from the priority queue while maintaining
        the heap property. This function handles batched operations efficiently,
        respecting the pop_ratio to control search width without losing nodes.

        This version performs EAGER EXPANSION and S' DEDUPLICATION.
        It expands (S, A) -> S' immediately inside the loop to ensure that the
        returned batch contains only unique S' states, preventing batch starvation
        in highly reversible environments.

        Args:
            search_result (SearchResult): The current search state
            puzzle (Puzzle): The puzzle instance
            solve_config (Puzzle.SolveConfig): Configuration for the puzzle

        Returns:
            tuple: Contains:
                - Updated SearchResult
                - A batch of the best values to be processed (Current)
                - A boolean mask indicating which entries in the batch are valid
        """

        def _expand_and_filter(search_result, key, val):
            """Helper to expand nodes, lookup in hashtable, and filter unoptimal ones."""
            filled = jnp.isfinite(key)
            parent_states = search_result.get_state(val.parent)
            parent_actions = val.parent.action  # [batch_size]
            parent_costs = search_result.get_cost(val.parent)

            current_states, ncosts = puzzle.batched_get_actions(
                solve_config, parent_states, parent_actions, filled
            )  # [batch_size] [action_size, batch_size]

            current_costs = parent_costs + ncosts
            current_dists = val.dist if use_heuristic else (val.dist - ncosts)

            # Optimization: deduplicate before lookup to avoid redundant hash lookups.
            unique_mask = xnp.unique_mask(current_states, current_costs, filled)
            current_hash_idxs, found = search_result.hashtable.lookup_parallel(
                current_states, unique_mask
            )

            old_costs = search_result.get_cost(current_hash_idxs)
            # Only consider nodes that are either:
            # 1. Not found in the hash table (new nodes), or
            # 2. Found but have better cost than existing
            better_cost_mask = jnp.less(current_costs, old_costs)
            optimal_mask = unique_mask & (jnp.logical_or(~found, better_cost_mask))

            # Mask unoptimal keys with infinity
            filtered_key = jnp.where(optimal_mask, key, jnp.inf)

            return current_states, current_costs, current_dists, filtered_key

        # 1. Get an initial batch from the Priority Queue (PQ)
        search_result.priority_queue, min_key, min_val = search_result.priority_queue.delete_mins()
        batch_size = search_result.batch_size
        stack_size = batch_size * 2

        # Initial expansion and filtering
        (current_states, current_costs, current_dists, min_key) = _expand_and_filter(
            search_result, min_key, min_val
        )

        # Apply unique mask to initial batch to maintain invariant
        filled = jnp.isfinite(min_key)
        unique_mask = xnp.unique_mask(current_states, current_costs, filled)
        min_key = jnp.where(unique_mask, min_key, jnp.inf)

        # Initialize buffer for overflow (inf)
        buffer_key = jnp.full((batch_size,), jnp.inf, dtype=KEY_DTYPE)
        buffer_val = Parant_with_Costs.default((batch_size,))

        # 2. Loop to fill the batch if it's not full of valid nodes
        def _cond(state):
            search_result, _, _, _, key, _, _, _ = state
            pq_not_empty = search_result.priority_queue.size > 0

            # Optimization: Stop if batch is mostly full (e.g. 90%)
            fill_ratio = jnp.mean(jnp.isfinite(key))
            not_full_enough = fill_ratio < POP_BATCH_FILLED_RATIO

            return jnp.logical_and(pq_not_empty, not_full_enough)

        def _body(state):
            search_result, current_states, costs, dists, key, val, _, _ = state

            # Pop new nodes from PQ
            (
                search_result.priority_queue,
                new_key,
                new_val,
            ) = search_result.priority_queue.delete_mins()

            # Expand and filter new nodes
            (new_states, new_costs, new_dists, new_key) = _expand_and_filter(
                search_result, new_key, new_val
            )

            # Merge with existing batch
            stack_states = xnp.concatenate((current_states, new_states))
            stack_costs = jnp.concatenate((costs, new_costs))
            stack_dists = jnp.concatenate((dists, new_dists))
            stack_key = jnp.concatenate((key, new_key))
            stack_val = xnp.concatenate((val, new_val))

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
            ),
        )

        final_parents = final_val.parent

        # Put overflow nodes back into the PQ so they are never lost
        search_result.priority_queue = jax.lax.cond(
            jnp.any(jnp.isfinite(overflow_keys)),
            lambda: search_result.priority_queue.insert(overflow_keys, overflow_vals),
            lambda: search_result.priority_queue,
        )

        # 3. Apply pop_ratio to the full batch
        filled = jnp.isfinite(final_key)
        # Add a small epsilon for floating point comparisons
        threshold = min_key[0] * search_result.pop_ratio + 1e-6

        # Identify nodes to process now vs. nodes to return to PQ
        process_mask = jnp.less_equal(min_key, threshold)
        base_process_mask = jnp.logical_and(filled, process_mask)

        # Enforce min_pop: ensure that we pop at least min_pop nodes if they are available.
        min_pop_mask = jnp.logical_and(jnp.cumsum(filled) <= search_result.min_pop, filled)

        # The final mask includes nodes within the threshold OR nodes to meet min_pop.
        final_process_mask = jnp.logical_or(base_process_mask, min_pop_mask)

        # Separate the nodes to be returned and re-insert them into the PQ
        return_keys = jnp.where(final_process_mask, jnp.inf, min_key)
        final_costs = jnp.where(final_process_mask, final_costs, jnp.inf)
        search_result.priority_queue = search_result.priority_queue.insert(return_keys, final_val)

        (
            search_result.hashtable,
            _,
            _,
            hash_idx,
        ) = search_result.hashtable.parallel_insert(final_states, final_process_mask)

        final_currents = Current(hashidx=hash_idx, cost=final_costs)

        search_result.cost = xnp.update_on_condition(
            search_result.cost,
            hash_idx.index,
            final_process_mask,
            final_costs,
        )
        search_result.dist = xnp.update_on_condition(
            search_result.dist,
            hash_idx.index,
            final_process_mask,
            final_dists,
        )
        search_result.parent = search_result.parent.at[hash_idx.index].set_as_condition(
            final_process_mask, final_parents
        )
        search_result.pop_generation = xnp.update_on_condition(
            search_result.pop_generation,
            hash_idx.index,
            final_process_mask,
            search_result.pop_count,
        )
        search_result.pop_count += 1

        if return_states:
            return search_result, final_currents, final_states, final_process_mask

        return search_result, final_currents, final_process_mask

    def get_solved_path(search_result) -> list[Parent]:
        """
        Get the path to the solved state.
        """
        assert search_result.solved
        solved_idx = search_result.solved_idx
        solved_index = int(jax.device_get(solved_idx.hashidx.index))
        max_steps = max(1, int(jax.device_get(search_result.generated_size)) + 1)

        (
            path_idx,
            path_costs,
            path_dists,
            length,
            loop_detected,
            corruption_detected,
        ) = _reconstruct_path_arrays(
            search_result.parent.hashidx.index,
            search_result.cost,
            search_result.dist,
            jnp.array(solved_index, dtype=jnp.int32),
            max_steps=max_steps,
        )
        path_idx = jax.device_get(path_idx)
        length = int(jax.device_get(length))
        loop_detected = bool(jax.device_get(loop_detected))
        corruption_detected = bool(jax.device_get(corruption_detected))

        path = [solved_idx]
        steps_added = 0
        for step in range(max(0, length - 1)):
            parent_idx = int(path_idx[step + 1])
            if parent_idx == -1:
                break
            path.append(search_result.parent[int(path_idx[step])])
            steps_added += 1

        path.reverse()

        if loop_detected:
            loop_idx = int(path_idx[length - 1]) if length > 0 else -1
            print(f"Loop detected in path reconstruction at index {loop_idx}")

        # Check that costs are monotonically increasing to prevent path corruption
        if corruption_detected:
            path_costs = jax.device_get(path_costs)
            path_dists = jax.device_get(path_dists)
            used_len = steps_added + 1
            costs = [float(val) for val in path_costs[:used_len]]
            dists = [float(val) for val in path_dists[:used_len]]
            costs.reverse()
            dists.reverse()
            print("ERROR: Path corruption detected - costs are not monotonically increasing!")
            print(f"costs: {costs}")
            print(f"dists: {dists}")
            for i in range(1, len(costs)):
                if costs[i] < costs[i - 1]:
                    raise AssertionError(
                        "Path corruption: costs are not monotonically increasing at index "
                        f"{i}: {costs[i-1]} > {costs[i]}"
                    )

        return path

    def get_state(search_result, idx: HashIdx | Current | Parent) -> Puzzle.State:
        """
        Get the state from the hash table.
        """
        if isinstance(idx, Current) or isinstance(idx, Parent):
            return search_result.hashtable[idx.hashidx]
        elif isinstance(idx, HashIdx):
            return search_result.hashtable[idx]
        else:
            raise ValueError(f"Invalid index type: {type(idx)}")

    def get_cost(search_result, idx: HashIdx | Current | Parent) -> chex.Array:
        """
        Get the cost of the state from the cost array.
        """
        if isinstance(idx, Current) or isinstance(idx, Parent):
            return search_result.cost[idx.hashidx.index]
        elif isinstance(idx, HashIdx):
            return search_result.cost[idx.index]
        else:
            raise ValueError(f"Invalid index type: {type(idx)}")

    def get_dist(search_result, idx: HashIdx | Current | Parent) -> chex.Array:
        """
        Get the distance of the state from the distance array.
        """
        if isinstance(idx, Current) or isinstance(idx, Parent):
            return search_result.dist[idx.hashidx.index]
        elif isinstance(idx, HashIdx):
            return search_result.dist[idx.index]
        else:
            raise ValueError(f"Invalid index type: {type(idx)}")

    def get_parent(search_result, idx: HashIdx | Current | Parent) -> Parent:
        """
        Get the parent action from the parent action array.
        """
        if isinstance(idx, Current) or isinstance(idx, Parent):
            return search_result.parent[idx.hashidx.index]
        elif isinstance(idx, HashIdx):
            return search_result.parent[idx.index]
        else:
            raise ValueError(f"Invalid index type: {type(idx)}")

    def mask_unoptimal(search_result, min_key: chex.Array, min_current: Current) -> chex.Array:
        """
        Mask the unoptimal states.
        """
        optimal = jnp.less_equal(min_current.cost, search_result.get_cost(min_current))
        return jnp.where(optimal, min_key, jnp.inf)


def unique_sort(
    key: chex.Array, val: Current, size: int | None = None
) -> tuple[chex.Array, Current]:
    """
    Sorts the keys and corresponding values.

    Args:
        key (chex.Array): Array of keys.
        val (Current): Array of values.
        size (int | None): Optional static length for the sort index array.

    Returns:
        tuple:
            - sorted_key (chex.Array): Sorted array of keys.
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
