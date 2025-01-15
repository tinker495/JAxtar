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
import jax.test_util

from JAxtar.annotate import (
    ACTION_DTYPE,
    HASH_POINT_DTYPE,
    HASH_TABLE_IDX_DTYPE,
    KEY_DTYPE,
)
from JAxtar.bgpq import BGPQ, HeapValue, bgpq_value_dataclass
from JAxtar.hash import HashTable
from JAxtar.util import set_array_as_condition, set_tree_as_condition
from puzzle.puzzle_base import Puzzle


@bgpq_value_dataclass
class HashTableidx_with_Parent_HeapValue:
    """
    A dataclass representing a hash table heap value for the priority queue.
    This class maintains the mapping between states in the hash table and their positions.

    Attributes:
        index (chex.Array): The index in the hash table where the state is stored
        table_index (chex.Array): The index of the cuckoo hash table (for collision resolution)
    """

    @bgpq_value_dataclass
    class Parent:
        index: chex.Array
        table_index: chex.Array
        action: chex.Array

        @staticmethod
        def default(shape=()) -> "HashTableidx_with_Parent_HeapValue.Parent":
            return HashTableidx_with_Parent_HeapValue.Parent(
                index=jnp.full(shape, -1, dtype=HASH_POINT_DTYPE),
                table_index=jnp.full(shape, -1, dtype=HASH_TABLE_IDX_DTYPE),
                action=jnp.full(shape, -1, dtype=ACTION_DTYPE),
            )

    @bgpq_value_dataclass
    class Current:
        index: chex.Array
        table_index: chex.Array
        cost: chex.Array

        @staticmethod
        def default(shape=()) -> "HashTableidx_with_Parent_HeapValue.Current":
            return HashTableidx_with_Parent_HeapValue.Current(
                index=jnp.full(shape, -1, dtype=HASH_POINT_DTYPE),
                table_index=jnp.full(shape, -1, dtype=HASH_TABLE_IDX_DTYPE),
                cost=jnp.full(shape, jnp.inf, dtype=KEY_DTYPE),
            )

    parent: Parent
    current: Current

    @staticmethod
    def default(shape=()) -> "HashTableidx_with_Parent_HeapValue":
        """
        Creates a default instance with -1 values, indicating an invalid/empty entry.

        Args:
            shape (tuple): The shape of the arrays to create

        Returns:
            HashTableidx_with_Parent_HeapValue: A new instance with default values
        """
        return HashTableidx_with_Parent_HeapValue(
            parent=HashTableidx_with_Parent_HeapValue.Parent.default(shape),
            current=HashTableidx_with_Parent_HeapValue.Current.default(shape),
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
    min_val_buffer: HashTableidx_with_Parent_HeapValue
    cost: chex.Array
    parent: chex.Array
    parent_action: chex.Array

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
        priority_queue = BGPQ.build(max_nodes, batch_size, HashTableidx_with_Parent_HeapValue)
        min_key_buffer = jnp.full((batch_size,), jnp.inf, dtype=KEY_DTYPE)
        min_val_buffer = HashTableidx_with_Parent_HeapValue.default((batch_size,))
        cost = jnp.full((size_table, n_table), jnp.inf, dtype=KEY_DTYPE)
        parent_action = jnp.full((size_table, n_table), -1, dtype=ACTION_DTYPE)
        parent = HashTableidx_with_Parent_HeapValue.Parent.default((size_table, n_table))
        return SearchResult(
            hashtable=hashtable,
            priority_queue=priority_queue,
            min_key_buffer=min_key_buffer,
            min_val_buffer=min_val_buffer,
            cost=cost,
            parent=parent,
            parent_action=parent_action,
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
    def size(self):
        """Current number of states stored."""
        return self.hashtable.size

    @property
    def min_states(self):
        """Minimum states in the priority queue."""
        min_val = self.priority_queue.val_store[0]  # get the minimum value
        return self.hashtable.table[min_val.current.index, min_val.current.table_index]

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
        search_result.priority_queue, min_key, min_val = BGPQ.delete_mins(
            search_result.priority_queue
        )
        min_val_cost = min_val.current.cost
        optimal = jnp.less(
            min_val_cost, search_result.cost[min_val.current.index, min_val.current.table_index]
        )
        min_key = jnp.where(optimal, min_key, jnp.inf)
        (
            min_key,
            min_val,
            search_result.min_key_buffer,
            search_result.min_val_buffer,
        ) = merge_sort_split(
            min_key, min_val, search_result.min_key_buffer, search_result.min_val_buffer
        )
        filled = jnp.isfinite(min_key)

        def _cond(val):
            search_result, _, _, filled = val
            return jnp.logical_and(search_result.priority_queue.size > 0, ~filled.all())

        def _body(val):
            search_result, min_key, min_val, filled = val
            search_result.priority_queue, new_key, new_val = BGPQ.delete_mins(
                search_result.priority_queue
            )
            new_val_cost = new_val.current.cost
            optimal = jnp.less(
                new_val_cost, search_result.cost[new_val.current.index, new_val.current.table_index]
            )
            new_key = jnp.where(optimal, new_key, jnp.inf)
            (
                min_key,
                min_val,
                search_result.min_key_buffer,
                search_result.min_val_buffer,
            ) = merge_sort_split(min_key, min_val, new_key, new_val)
            filled = jnp.isfinite(min_key)
            return search_result, min_key, min_val, filled

        search_result, min_key, min_val, filled = jax.lax.while_loop(
            _cond, _body, (search_result, min_key, min_val, filled)
        )
        search_result.cost = set_array_as_condition(
            search_result.cost,
            filled,
            min_val.current.cost,
            min_val.current.index,
            min_val.current.table_index,
        )
        search_result.parent = set_tree_as_condition(
            search_result.parent,
            filled,
            min_val.parent,
            min_val.current.index,
            min_val.current.table_index,
        )
        search_result.parent_action = set_array_as_condition(
            search_result.parent_action,
            filled,
            min_val.parent.action,
            min_val.current.index,
            min_val.current.table_index,
        )
        return search_result, min_val.current, filled


def unique_mask(val: HashTableidx_with_Parent_HeapValue, batch_len: int):
    """
    Creates a boolean mask identifying unique values in a HashTableIdx_HeapValue tensor.
    This function is used to filter out duplicate states in batched operations.

    Args:
        val (HashTableIdx_HeapValue): The heap values to check for uniqueness
        batch_len (int): The length of the batch

    Returns:
        jnp.ndarray: Boolean mask where True indicates unique values
    """
    min_val_stack = jnp.stack([val.current.index, val.current.table_index], axis=1)
    unique_idxs = jnp.unique(min_val_stack, axis=0, size=batch_len, return_index=True)[
        1
    ]  # val = (unique_len, 2), unique_idxs = (unique_len,)
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
