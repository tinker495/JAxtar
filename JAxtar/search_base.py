"""
JAxtar Search Base Module
This module implements the core search functionality for A*/Q* algorithms using JAX.
The implementation is designed to be fully parallelizable and GPU-compatible.
Key features:
- Pure JAX implementation for ML research
- Batched operations for GPU optimization
- Generic puzzle-agnostic implementation
"""

import chex
import jax
import jax.numpy as jnp

from JAxtar.annotate import (
    ACTION_DTYPE,
    HASH_POINT_DTYPE,
    HASH_TABLE_IDX_DTYPE,
    KEY_DTYPE,
)
from JAxtar.bgpq import BGPQ, HeapValue, bgpq_value_dataclass
from JAxtar.hash import HashTable
from puzzle.puzzle_base import Puzzle


@bgpq_value_dataclass
class HashTableIdx_HeapValue:
    """
    A dataclass representing a hash table heap value for the priority queue.
    This class maintains the mapping between states in the hash table and their positions.

    Attributes:
        index (chex.Array): The index in the hash table where the state is stored
        table_index (chex.Array): The index of the cuckoo hash table (for collision resolution)
    """

    index: chex.Array
    table_index: chex.Array

    @staticmethod
    def default(shape=()) -> "HashTableIdx_HeapValue":
        """
        Creates a default instance with -1 values, indicating an invalid/empty entry.

        Args:
            shape (tuple): The shape of the arrays to create

        Returns:
            HashTableIdx_HeapValue: A new instance with default values
        """
        return HashTableIdx_HeapValue(
            index=jnp.full(shape, -1, dtype=HASH_POINT_DTYPE),
            table_index=jnp.full(shape, -1, dtype=HASH_TABLE_IDX_DTYPE),
        )


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
        min_val_buffer (HashTableIdx_HeapValue): Buffer for minimum values in the priority queue
        cost (chex.Array): Cost array tracking path costs to each state
        not_closed (chex.Array): Boolean array tracking open/closed status (inverted for efficiency)
        parent (chex.Array): Array storing parent state indices for path reconstruction
        parent_action (chex.Array): Array storing actions that led to each state
    """

    hashtable: HashTable
    priority_queue: BGPQ
    min_key_buffer: chex.Array
    min_val_buffer: HashTableIdx_HeapValue
    cost: chex.Array
    not_closed: chex.Array
    parent: chex.Array
    parent_action: chex.Array
    solved: chex.Array
    solved_idx: HashTableIdx_HeapValue

    @staticmethod
    def build(statecls: Puzzle.State, batch_size: int, max_nodes: int, seed=0, n_table=2):
        """
        Creates a new instance of SearchResult with initialized data structures.

        Args:
            statecls (Puzzle.State): The state class for the puzzle being solved
            batch_size (int): Size of batches for parallel processing
            max_nodes (int): Maximum number of nodes to store
            seed (int): Random seed for hash function initialization
            n_table (int): Number of cuckoo hash tables for collision handling

        Returns:
            SearchResult: A new instance with initialized data structures
        """
        # Initialize the hash table for state storage
        hashtable = HashTable.build(statecls, seed, max_nodes, n_table=n_table)
        size_table = hashtable.capacity
        n_table = hashtable.n_table

        # Initialize priority queue for state expansion
        priority_queue = BGPQ.build(max_nodes, batch_size, HashTableIdx_HeapValue)

        # Initialize buffers for minimum values
        min_key_buffer = jnp.full((batch_size,), jnp.inf, dtype=KEY_DTYPE)
        min_val_buffer = HashTableIdx_HeapValue.default((batch_size,))

        # Initialize arrays for tracking costs and state relationships
        cost = jnp.full((size_table, n_table), jnp.inf, dtype=KEY_DTYPE)
        not_closed = jnp.ones((size_table, n_table), dtype=jnp.bool)
        parent = HashTableIdx_HeapValue.default((size_table, n_table))
        parent_action = jnp.full((size_table, n_table), -1, dtype=ACTION_DTYPE)
        solved = jnp.array(False)
        solved_idx = HashTableIdx_HeapValue.default((1,))

        return SearchResult(
            hashtable=hashtable,
            priority_queue=priority_queue,
            min_key_buffer=min_key_buffer,
            min_val_buffer=min_val_buffer,
            cost=cost,
            not_closed=not_closed,
            parent=parent,
            parent_action=parent_action,
            solved=solved,
            solved_idx=solved_idx,
        )

    @property
    def capacity(self):
        """Maximum number of states that can be stored."""
        return self.hashtable.capacity

    @property
    def n_table(self):
        """Number of cuckoo hash tables being used."""
        return self.hashtable.n_table

    @property
    def batch_size(self):
        """Batch size of the search."""
        return self.priority_queue.batch_size

    @property
    def size(self):
        """Current number of states stored."""
        return self.hashtable.size

    def pop_full(search_result):
        """
        Removes and returns the minimum elements from the priority queue while maintaining
        the heap property. This function handles batched operations efficiently.

        Args:
            search_result (SearchResult): The current search state

        Returns:
            tuple: Contains:
                - Updated SearchResult
                - Minimum values removed from the queue
                - Boolean mask indicating which entries were filled
        """
        # Delete minimum elements from the priority queue
        search_result.priority_queue, min_key, min_val = search_result.priority_queue.delete_mins()

        # Check if the states are in the open set
        not_closed = search_result.not_closed[min_val.index, min_val.table_index]
        min_key = jnp.where(not_closed, min_key, jnp.inf)  # Set closed states to inf

        # Merge and sort with the buffer
        (
            min_key,
            min_val,
            search_result.min_key_buffer,
            search_result.min_val_buffer,
        ) = merge_sort_split(
            search_result.min_key_buffer, search_result.min_val_buffer, min_key, min_val
        )
        filled = jnp.isfinite(min_key)

        def _cond(val):
            """Check if we need to continue popping elements."""
            search_result, _, _, filled = val
            cond1 = search_result.priority_queue.size > 0  # if queue is empty, we are done
            cond2 = ~filled.all()  # if all states are filled, we are done
            return jnp.logical_and(cond1, cond2)

        def _body(val):
            """Process one batch of elements from the priority queue."""
            search_result, min_key, min_val, filled = val
            (
                search_result.priority_queue,
                new_key,
                new_val,
            ) = search_result.priority_queue.delete_mins()
            not_closed = search_result.not_closed[new_val.index, new_val.table_index]
            new_key = jnp.where(not_closed, new_key, jnp.inf)

            # Merge new values with current minimum values
            # if filled is not all true, min buffer is will be empty(filled with inf keys)
            (
                min_key,
                min_val,
                search_result.min_key_buffer,
                search_result.min_val_buffer,
            ) = merge_sort_split(min_key, min_val, new_key, new_val)
            filled = jnp.isfinite(min_key)
            return search_result, min_key, min_val, filled

        # Continue popping elements until we have enough or queue is empty
        search_result, min_key, min_val, filled = jax.lax.while_loop(
            _cond, _body, (search_result, min_key, min_val, filled)
        )

        # Update the closed set
        search_result.not_closed = search_result.not_closed.at[
            min_val.index, min_val.table_index
        ].set(~filled)
        return search_result, min_val, filled

    def get_solved_path(search_result):
        """
        Get the path to the solved state.
        """
        assert search_result.solved
        parents = search_result.parent
        solved_idx = search_result.solved_idx

        path = [solved_idx]
        parent_last = parents[solved_idx.index, solved_idx.table_index]
        while True:
            if parent_last.index == -1:
                break
            path.append(parent_last)
            parent_last = parents[parent_last.index, parent_last.table_index]
        path.reverse()
        return path

    def get_state(search_result, idx: HashTableIdx_HeapValue):
        """
        Get the state from the hash table.
        """
        return search_result.hashtable.table[idx.index, idx.table_index]

    def get_cost(search_result, idx: HashTableIdx_HeapValue):
        """
        Get the cost of the state from the cost array.
        """
        return search_result.cost[idx.index, idx.table_index]

    def get_parent_action(search_result, idx: HashTableIdx_HeapValue):
        """
        Get the parent action from the parent action array.
        """
        return search_result.parent_action[idx.index, idx.table_index]


def unique_mask(val: HashTableIdx_HeapValue, batch_len: int):
    """
    Creates a boolean mask identifying unique values in a HashTableIdx_HeapValue tensor.
    This function is used to filter out duplicate states in batched operations.

    Args:
        val (HashTableIdx_HeapValue): The heap values to check for uniqueness
        batch_len (int): The length of the batch

    Returns:
        jnp.ndarray: Boolean mask where True indicates unique values
    """
    min_val_stack = jnp.stack([val.index, val.table_index], axis=1)
    unique_idxs = jnp.unique(min_val_stack, axis=0, size=batch_len, return_index=True)[1]
    uniques = jnp.zeros((batch_len,), dtype=jnp.bool_).at[unique_idxs].set(True)
    return uniques


def merge_sort_split(
    ak: chex.Array, av: HeapValue, bk: chex.Array, bv: HeapValue
) -> tuple[chex.Array, HeapValue, chex.Array, HeapValue]:
    """
    Merges and sorts two key-value pairs, then splits them back into two equal parts.
    This operation is crucial for maintaining the heap property in the priority queue.

    Args:
        ak (chex.Array): First array of keys
        av (HeapValue): First array of values
        bk (chex.Array): Second array of keys
        bv (HeapValue): Second array of values

    Returns:
        tuple: Contains:
            - First half of sorted keys
            - First half of corresponding values
            - Second half of sorted keys
            - Second half of corresponding values
    """
    n = ak.shape[-1]  # size of group
    key = jnp.concatenate([ak, bk])
    val = jax.tree_util.tree_map(lambda a, b: jnp.concatenate([a, b]), av, bv)

    uniques = unique_mask(val, 2 * n)
    key = jnp.where(uniques, key, jnp.inf)  # Set duplicate keys to inf to ensure they sort last

    idx = jnp.argsort(key, stable=True)
    sorted_key = key[idx]
    sorted_val = jax.tree_util.tree_map(lambda x: x[idx], val)
    return sorted_key[:n], sorted_val[:n], sorted_key[n:], sorted_val[n:]
