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


@xtructure_dataclass
class Parent:

    hashidx: FieldDescriptor.scalar(dtype=HashIdx)
    action: FieldDescriptor.scalar(dtype=ACTION_DTYPE)


@xtructure_dataclass
class Current:

    hashidx: FieldDescriptor.scalar(dtype=HashIdx)
    cost: FieldDescriptor.scalar(dtype=KEY_DTYPE)


@xtructure_dataclass
class Current_with_Parent:
    """
    A dataclass representing a hash table heap value for the priority queue.
    This class maintains the mapping between states in the hash table and their positions.

    Attributes:
        parent (Parent): The parent state in the search tree
        current (Current): The current state in the search tree
    """

    parent: FieldDescriptor.scalar(dtype=Parent)
    current: FieldDescriptor.scalar(dtype=Current)


@xtructure_dataclass
class Parant_with_Costs:
    """
    A dataclass representing a hash table heap value for the priority queue.
    This class maintains the mapping between states in the hash table and their positions.
    """

    parent: FieldDescriptor.scalar(dtype=Parent)
    cost: FieldDescriptor.scalar(dtype=KEY_DTYPE)
    dist: FieldDescriptor.scalar(dtype=KEY_DTYPE)


@base_dataclass
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
    @partial(jax.jit, static_argnums=(0, 1, 2), static_argnames=("parant_with_costs",))
    def build(
        statecls: Puzzle.State,
        batch_size: int,
        max_nodes: int,
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
            priority_queue = BGPQ.build(max_nodes, batch_size, Current_with_Parent, KEY_DTYPE)

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

    @property
    def opened_size(self) -> int:
        """Current number of states stored in the opened set.
        This is the number of states in the hash table but not in the closed set."""
        return self.generated_size - self.closed_size

    @property
    def closed_size(self) -> int:
        """Current number of states stored in the closed set.
        This is the number of states in the closed set."""
        return jnp.sum(jnp.isfinite(self.cost))

    def pop_full(search_result) -> tuple["SearchResult", Current, chex.Array]:
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
        min_key = search_result.mask_unoptimal(min_key, min_val.current)
        buffer = jnp.full(min_key.shape, jnp.inf, dtype=KEY_DTYPE)
        buffer_val = Current_with_Parent.default(min_key.shape)

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
            new_key = search_result.mask_unoptimal(new_key, new_val.current)

            # Merge current batch with new nodes, splitting into main and overflow
            main_keys, main_vals, overflow_keys, overflow_vals = _unique_sort_merge_and_split(
                key, val, new_key, new_val
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

        # 4. Update the closed set with the nodes we are processing
        search_result.cost = xnp.update_on_condition(
            search_result.cost,
            min_val.current.hashidx.index,
            final_process_mask,
            min_val.current.cost,
        )
        search_result.parent = search_result.parent.at[
            min_val.current.hashidx.index
        ].set_as_condition(final_process_mask, min_val.parent)
        search_result.pop_generation = xnp.update_on_condition(
            search_result.pop_generation,
            min_val.current.hashidx.index,
            final_process_mask,
            search_result.pop_count,
        )
        search_result.pop_count += 1

        return search_result, min_val.current, final_process_mask

    def pop_full_with_actions(
        search_result, puzzle: Puzzle, solve_config: Puzzle.SolveConfig
    ) -> tuple["SearchResult", Current, chex.Array, chex.Array]:
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
                - Next states (S')
                - A boolean mask indicating which entries in the batch are valid
        """

        def _expand_and_filter(search_result, key, val):
            """Helper to expand nodes, lookup in hashtable, and filter unoptimal ones."""
            filled = jnp.isfinite(key)
            parent_states = search_result.get_state(val.parent)
            parent_actions = val.parent.action
            parent_costs = search_result.get_cost(val.parent)

            current_states, ncosts = puzzle.batched_get_actions(
                solve_config, parent_states, parent_actions, filled
            )
            current_costs = parent_costs + ncosts
            current_dists = val.dist - ncosts

            # Optimization: deduplicate before lookup could save compute,
            # but here we stick to logic that filters after lookup to check against global best.
            current_hash_idxs, found = search_result.hashtable.lookup_parallel(current_states)

            old_costs = search_result.get_cost(current_hash_idxs)
            # Only consider nodes that are either:
            # 1. Not found in the hash table (new nodes), or
            # 2. Found but have better cost than existing
            better_cost_mask = jnp.less(current_costs, old_costs)
            optimal_mask = (jnp.logical_or(~found, better_cost_mask)) & filled

            # Mask unoptimal keys with infinity
            filtered_key = jnp.where(optimal_mask, key, jnp.inf)

            return current_states, current_costs, current_dists, filtered_key

        # 1. Get an initial batch from the Priority Queue (PQ)
        search_result.priority_queue, min_key, min_val = search_result.priority_queue.delete_mins()
        batch_size = min_key.shape[-1]

        # Initial expansion and filtering
        (current_states, current_costs, current_dists, min_key) = _expand_and_filter(
            search_result, min_key, min_val
        )

        # Apply unique mask to initial batch to maintain invariant
        filled = jnp.isfinite(min_key)
        unique_mask = xnp.unique_mask(current_states, current_costs, filled)
        min_key = jnp.where(unique_mask, min_key, jnp.inf)

        # Initialize buffer for overflow (inf)
        buffer_key = jnp.full(min_key.shape, jnp.inf, dtype=KEY_DTYPE)
        buffer_val = Parant_with_Costs.default(min_key.shape)

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
            sorted_key, sorted_idx = jax.lax.sort_key_val(
                stack_key, jnp.arange(stack_key.shape[-1])
            )
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

        return search_result, final_currents, final_states, final_process_mask

    def get_solved_path(search_result) -> list[Parent]:
        """
        Get the path to the solved state.
        """
        assert search_result.solved
        solved_idx = search_result.solved_idx

        path = [solved_idx]
        parent_last = search_result.get_parent(solved_idx)
        visited = set()
        while True:
            idx = int(parent_last.hashidx.index)
            if parent_last.hashidx.index == -1:
                break
            if idx in visited:
                print(f"Loop detected in path reconstruction at index {idx}")
                break
            visited.add(idx)
            path.append(parent_last)
            parent_last = search_result.get_parent(parent_last)
        path.reverse()
        return path

    def get_state(
        search_result, idx: HashIdx | Current | Parent | Current_with_Parent
    ) -> Puzzle.State:
        """
        Get the state from the hash table.
        """
        if isinstance(idx, (Current_with_Parent)):
            return search_result.hashtable[idx.current.hashidx]
        elif isinstance(idx, Current) or isinstance(idx, Parent):
            return search_result.hashtable[idx.hashidx]
        elif isinstance(idx, HashIdx):
            return search_result.hashtable[idx]
        else:
            raise ValueError(f"Invalid index type: {type(idx)}")

    def get_cost(
        search_result, idx: HashIdx | Current | Parent | Current_with_Parent
    ) -> chex.Array:
        """
        Get the cost of the state from the cost array.
        """
        if isinstance(idx, (Current_with_Parent, Parant_with_Costs)):
            return search_result.cost[idx.current.hashidx.index]
        elif isinstance(idx, Current) or isinstance(idx, Parent):
            return search_result.cost[idx.hashidx.index]
        elif isinstance(idx, HashIdx):
            return search_result.cost[idx.index]
        else:
            raise ValueError(f"Invalid index type: {type(idx)}")

    def get_dist(
        search_result, idx: HashIdx | Current | Parent | Current_with_Parent
    ) -> chex.Array:
        """
        Get the distance of the state from the distance array.
        """
        if isinstance(idx, (Current_with_Parent)):
            return search_result.dist[idx.current.hashidx.index]
        elif isinstance(idx, Current) or isinstance(idx, Parent):
            return search_result.dist[idx.hashidx.index]
        elif isinstance(idx, HashIdx):
            return search_result.dist[idx.index]
        else:
            raise ValueError(f"Invalid index type: {type(idx)}")

    def get_parent(search_result, idx: HashIdx | Current | Parent | Current_with_Parent) -> Parent:
        """
        Get the parent action from the parent action array.
        """
        if isinstance(idx, (Current_with_Parent)):
            return search_result.parent[idx.current.hashidx.index]
        elif isinstance(idx, Current) or isinstance(idx, Parent):
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
    key: chex.Array, val: Current_with_Parent
) -> tuple[chex.Array, Current_with_Parent]:
    """
    Sorts the keys and corresponding values.

    Args:
        key (chex.Array): Array of keys.
        val (Current_with_Parent): Array of values.

    Returns:
        tuple:
            - sorted_key (chex.Array): Sorted array of keys.
            - sorted_val (Current_with_Parent): Values corresponding to the sorted keys.
    """
    n = key.shape[-1]
    filled = jnp.isfinite(key)
    mask = xnp.unique_mask(val.current.hashidx, val.current.cost, filled)
    key = jnp.where(mask, key, jnp.inf)
    sorted_key, sorted_idx = jax.lax.sort_key_val(key, jnp.arange(n))
    sorted_val = val[sorted_idx]
    return sorted_key, sorted_val


# Helper to merge, sort, and split two batches of nodes
def _unique_sort_merge_and_split(k1, v1, k2, v2):
    batch_size = k1.shape[-1]
    merged_key = jnp.concatenate([k1, k2])
    merged_val = xnp.concatenate([v1, v2])

    # Sort and remove duplicates from the combined batch
    sorted_key, sorted_val = unique_sort(merged_key, merged_val)

    # Split into a main batch and an overflow batch
    main_keys = sorted_key[:batch_size]
    main_vals = sorted_val[:batch_size]
    overflow_keys = sorted_key[batch_size:]
    overflow_vals = sorted_val[batch_size:]
    return main_keys, main_vals, overflow_keys, overflow_vals
