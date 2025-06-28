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

from JAxtar.annotate import (
    ACTION_DTYPE,
    CUCKOO_TABLE_N,
    HASH_SIZE_MULTIPLIER,
    KEY_DTYPE,
)
from JAxtar.util import set_array_as_condition


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
    min_key_buffer: chex.Array  # buffer for minimum keys
    min_val_buffer: Xtructurable | Current_with_Parent  # buffer for minimum values
    cost: chex.Array  # cost array - g value
    dist: chex.Array  # distance array - calculated heuristic or Q value
    parent: Xtructurable | Parent  # parent array
    solved: chex.Array  # solved array
    solved_idx: Xtructurable | Current  # solved index

    @staticmethod
    @partial(jax.jit, static_argnums=(0, 1, 2))
    def build(statecls: Puzzle.State, batch_size: int, max_nodes: int, seed=42):
        """
        Creates a new instance of SearchResult with initialized data structures.

        Args:
            statecls (Puzzle.State): The state class for the puzzle being solved
            batch_size (int): Size of batches for parallel processing
            max_nodes (int): Maximum number of nodes to store
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

        # Initialize buffers for minimum values
        min_key_buffer = jnp.full((batch_size,), jnp.inf, dtype=KEY_DTYPE)
        min_val_buffer = Current_with_Parent.default((batch_size,))

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
            min_key_buffer=min_key_buffer,
            min_val_buffer=min_val_buffer,
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
        min_val_cost = min_val.current.cost
        optimal = jnp.less(min_val_cost, search_result.get_cost(min_val.current))
        min_key = jnp.where(optimal, min_key, jnp.inf)  # Set closed states to inf

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
            new_val_cost = new_val.current.cost
            optimal = jnp.less(new_val_cost, search_result.get_cost(new_val.current))
            new_key = jnp.where(optimal, new_key, jnp.inf)

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
        search_result.cost = set_array_as_condition(
            search_result.cost,
            filled,
            min_val.current.cost,
            min_val.current.hashidx.index,
        )
        search_result.parent = search_result.parent.at[
            min_val.current.hashidx.index
        ].set_as_condition(filled, min_val.parent)
        return search_result, min_val.current, filled

    def get_solved_path(search_result) -> list[Parent]:
        """
        Get the path to the solved state.
        """
        assert search_result.solved
        solved_idx = search_result.solved_idx

        path = [solved_idx]
        parent_last = search_result.get_parent(solved_idx)
        while True:
            if parent_last.hashidx.index == -1:
                break
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


def unique_mask(val: Current_with_Parent, batch_len: int) -> chex.Array:
    """
    Creates a boolean mask identifying the first occurrence of unique values in a
    Current_with_Parent pytree. It is robust to padding from jnp.unique.

    Args:
        val (Current_with_Parent): The pytree of values to check for uniqueness.
        batch_len (int): The length of the batch dimension of the pytree.

    Returns:
        jnp.ndarray: Boolean mask where True indicates the first unique value.
    """
    hash_idx_bytes = jax.vmap(lambda x: x.current.hashidx.uint32ed)(val)
    _, unique_indices = jnp.unique(hash_idx_bytes, axis=0, return_index=True, size=batch_len)
    mask = jnp.zeros((batch_len,), dtype=jnp.bool_).at[unique_indices].set(True)
    return mask


def merge_sort_split(
    ak: chex.Array, av: Current_with_Parent, bk: chex.Array, bv: Current_with_Parent
) -> tuple[chex.Array, Current_with_Parent, chex.Array, Current_with_Parent]:
    """
    Merges and sorts two key-value pairs, then splits them back into two equal parts.
    This operation is crucial for maintaining the heap property in the priority queue.
    This implementation correctly handles duplicates by ensuring the entry with the
    minimum key is kept, preserving optimality.

    Args:
        ak (chex.Array): First array of keys
        av (Current_with_Parent): First array of values
        bk (chex.Array): Second array of keys
        bv (Current_with_Parent): Second array of values

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

    mask = unique_mask(val, 2 * n)

    # Invalidate the keys of all non-optimal duplicates.
    key_with_invalids = jnp.where(mask, key, jnp.inf)

    # Final sort to compact the array by moving invalid keys to the end.
    sorted_key, sorted_idx = jax.lax.sort_key_val(key_with_invalids, jnp.arange(2 * n))
    sorted_val = val[sorted_idx]

    return sorted_key[:n], sorted_val[:n], sorted_key[n:], sorted_val[n:]
