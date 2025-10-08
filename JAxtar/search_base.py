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
from typing import Union

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


@xtructure_dataclass
class Parent:

    hashidx: FieldDescriptor[HashIdx]
    action: FieldDescriptor[ACTION_DTYPE]


@xtructure_dataclass
class Current:

    hashidx: FieldDescriptor[HashIdx]
    cost: FieldDescriptor[KEY_DTYPE]


@xtructure_dataclass
class Current_with_Parent:
    """
    A dataclass representing a hash table heap value for the priority queue.
    This class maintains the mapping between states in the hash table and their positions.

    Attributes:
        parent (Parent): The parent state in the search tree
        current (Current): The current state in the search tree
    """

    parent: FieldDescriptor[Parent]
    current: FieldDescriptor[Current]


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
    @partial(jax.jit, static_argnums=(0, 1, 2))
    def build(
        statecls: Puzzle.State,
        batch_size: int,
        max_nodes: int,
        pop_ratio: float = jnp.inf,
        min_pop: int = 1,
        seed=42,
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

        # Initialize priority queue for state expansion
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

        # 1. Get an initial batch from the Priority Queue (PQ)
        search_result.priority_queue, min_key, min_val = search_result.priority_queue.delete_mins()
        min_key = search_result.mask_unoptimal(min_key, min_val)
        buffer = jnp.full(min_key.shape, jnp.inf, dtype=KEY_DTYPE)
        buffer_val = Current_with_Parent.default(min_key.shape)

        # 2. Loop to fill the batch if it's not full of valid nodes
        def _cond(state):
            search_result, key, _, _, _ = state
            pq_not_empty = search_result.priority_queue.size > 0
            batch_has_empty_slots = jnp.isinf(key).any()

            # For pop_ratio = inf, we don't check threshold and fill the batch completely.
            # For other cases, we can stop early, but the final filtering is done later.
            # This loop is mainly for ensuring we have enough candidates to select from.
            return jnp.logical_and(pq_not_empty, batch_has_empty_slots)

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

    def get_solved_path(search_result) -> list[Union[Parent, Current]]:
        """
        Get the path to the solved state.

        returns:
            path: list[Parent, Current] - [Parent, Parent, ..., Parent, Current]
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

    @partial(jax.jit, static_argnums=(3,))
    def _get_path(
        search_result, solved_idx: Current, mask: chex.Array = True, max_depth: int = 100
    ) -> tuple[Parent, chex.Array]:
        """
        Get the path to the solved state using jax.lax.scan for JIT compatibility.

        Args:
            search_result: The search result containing parent information
            solved_idx: The index of the solved state
            max_depth: Maximum depth of the path to return

        Returns:
            tuple: Contains:
                - Array of parents with fixed length max_depth
                - Boolean mask indicating valid entries in the path
        """
        parent = search_result.get_parent(solved_idx)

        # Use jax.lax.scan to collect parents
        def scan_fn(parent, _):
            cont = jnp.logical_and(parent.hashidx.index != -1, mask)
            next_parent = jax.lax.cond(
                cont, lambda: search_result.get_parent(parent), lambda: parent
            )
            return next_parent, (parent, cont)

        _, (path, path_mask) = jax.lax.scan(scan_fn, parent, length=max_depth)
        return path, path_mask

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

    def mask_unoptimal(
        search_result, min_key: chex.Array, min_val: Current_with_Parent
    ) -> chex.Array:
        """
        Mask the unoptimal states.
        """
        optimal = jnp.less_equal(min_val.current.cost, search_result.get_cost(min_val.current))
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
