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
from puxle import Puzzle
from xtructure import (
    BGPQ,
    FieldDescriptor,
    HashIdx,
    HashTable,
    Xtructurable,
    xtructure_dataclass,
)
from xtructure import xtructure_numpy as xnp

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


@chex.dataclass
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
        min_key_buffer (chex.Array): Buffer for minimum keys in the priority queue
        min_val_buffer (Current_with_Parent): Buffer for minimum values in the priority queue
        cost (chex.Array): Cost array tracking path costs to each state (g value)
        dist (chex.Array): Distance array storing calculated heuristic or Q values
        parent (Parent): Array storing parent state indices for path reconstruction
        solved (chex.Array): Boolean flag indicating if a solution has been found
        solved_idx (Current): Index of the solved state in the hash table
    """

    hashtable: HashTable  # hash table
    priority_queue: BGPQ  # priority queue
    pop_ratio: float  # ratio of states to pop from the priority queue
    cost: chex.Array  # cost array - g value
    dist: chex.Array  # distance array - calculated heuristic or Q value
    parent: Xtructurable | Parent  # parent array
    solved: chex.Array  # solved array
    solved_idx: Xtructurable | Current  # solved index

    @staticmethod
    @partial(jax.jit, static_argnums=(0, 1, 2))
    def build(
        statecls: Puzzle.State, batch_size: int, max_nodes: int, pop_ratio: float = jnp.inf, seed=42
    ):
        """
        Creates a new instance of SearchResult with initialized data structures.

        Args:
            statecls (Puzzle.State): The state class for the puzzle being solved
            batch_size (int): Size of batches for parallel processing
            max_nodes (int): Maximum number of nodes to store
            pop_ratio (float): Controls the search beam width. Nodes are expanded if their cost is
                within `pop_ratio` percent of the best node's cost. For instance, 0.1 allows for a
                10% margin. A value of 'inf' corresponds to a fixed-width beam search determined
                by the batch size.
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

        return SearchResult(
            hashtable=hashtable,
            priority_queue=priority_queue,
            pop_ratio=jnp.maximum(1.0 + pop_ratio, 1.1),  # minimum 10% margin
            cost=cost,
            dist=dist,
            parent=parent,
            solved=solved,
            solved_idx=solved_idx,
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

        # 2. Loop to fill the batch if it's not full of valid nodes
        def _cond(state):
            search_result, key, _ = state
            pq_not_empty = search_result.priority_queue.size > 0
            batch_has_empty_slots = jnp.isinf(key).any()

            # Early exit if a node exceeds the pop_ratio threshold
            best_key = key[0]
            threshold = best_key * search_result.pop_ratio + 1e-6
            finite_keys_mask = jnp.isfinite(key)
            exceeds_threshold = jnp.any(jnp.where(finite_keys_mask, key, -jnp.inf) > threshold)

            # Continue if PQ is not empty, batch has slots, and no node has exceeded the threshold
            return jnp.logical_and(
                pq_not_empty,
                jnp.logical_and(batch_has_empty_slots, jnp.logical_not(exceeds_threshold)),
            )

        def _body(state):
            search_result, key, val = state
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

            # Put overflow nodes back into the PQ so they are never lost
            search_result.priority_queue = search_result.priority_queue.insert(
                overflow_keys, overflow_vals
            )

            return search_result, main_keys, main_vals

        # Run the loop until we have a full batch of the best available nodes
        search_result, min_key, min_val = jax.lax.while_loop(
            _cond, _body, (search_result, min_key, min_val)
        )

        # 3. Apply pop_ratio to the full batch
        filled = jnp.isfinite(min_key)
        # Add a small epsilon for floating point comparisons
        threshold = min_key[0] * search_result.pop_ratio + 1e-6

        # Identify nodes to process now vs. nodes to return to PQ
        process_mask = jnp.less_equal(min_key, threshold)
        final_process_mask = jnp.logical_and(filled, process_mask)

        # Separate the nodes to be returned and re-insert them into the PQ
        return_keys = jnp.where(final_process_mask, jnp.inf, min_key)
        search_result.priority_queue = search_result.priority_queue.insert(return_keys, min_val)

        # 4. Update the closed set with the nodes we are processing
        search_result.cost = xnp.set_as_condition_on_array(
            search_result.cost,
            min_val.current.hashidx.index,
            final_process_mask,
            min_val.current.cost,
        )
        search_result.parent = search_result.parent.at[
            min_val.current.hashidx.index
        ].set_as_condition(final_process_mask, min_val.parent)

        return search_result, min_val.current, final_process_mask

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
    mask = xnp.unique_mask(val.current.hashidx, val.current.cost)
    key = jnp.where(mask, key, jnp.inf)
    sorted_key, sorted_idx = jax.lax.sort_key_val(key, jnp.arange(n))
    sorted_val = val[sorted_idx]
    return sorted_key, sorted_val
