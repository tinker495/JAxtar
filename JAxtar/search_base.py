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
    This class is a dataclass that represents a hash table heap value.
    It has two fields:
    1. index: hashtable index
    2. table_index: cuckoo table index
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
        return HashTableidx_with_Parent_HeapValue(
            parent=HashTableidx_with_Parent_HeapValue.Parent.default(shape),
            current=HashTableidx_with_Parent_HeapValue.Current.default(shape),
        )


@chex.dataclass
class SearchResult:
    """
    SearchResult is a dataclass that contains the data structures used in the A*/Q* algorithm.

    Note:
    - opened set: not in closed set, this could be not in HashTable or in HashTable but not in closed set.
    - closed set: available at HashTable, and in closed set.

    Attributes:
    - hashtable: HashTable instance that contains the states.
    - priority_queue: BGPQ instance that contains the indexes of the states in the HashTable.
    - cost: cost of the path from the start node to the current node.
            this could be update if a better path is found.
    - not_closed: a boolean array that indicates whether the state is in the closed set or not.
                this is inverted for the efficient implementation. not_closed = ~closed
    - parent: a 2D array that contains the index of the parent node.
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
        build is a static method that creates a new instance of AstarResult.
        """
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
        return self.hashtable.capacity

    @property
    def n_table(self):
        return self.hashtable.n_table

    @property
    def size(self):
        return self.hashtable.size


def unique_mask(val: HashTableidx_with_Parent_HeapValue, batch_len: int):
    """
    unique_mask is a function that returns a boolean mask of the unique values in the val tensor.
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
    Merge two sorted key tensors ak and bk as well as corresponding
    value tensors av and bv into a single sorted tensor.

    Args:
        ak: chex.Array - sorted key tensor
        av: HeapValue - sorted value tensor
        bk: chex.Array - sorted key tensor
        bv: HeapValue - sorted value tensor

    Returns:
        key1: chex.Array - merged and sorted
        val1: HeapValue - merged and sorted
        key2: chex.Array - merged and sorted
        val2: HeapValue - merged and sorted
    """
    n = ak.shape[-1]  # size of group
    key = jnp.concatenate([ak, bk])
    val = jax.tree_util.tree_map(lambda a, b: jnp.concatenate([a, b]), av, bv)

    uniques = unique_mask(val, 2 * n)
    key = jnp.where(uniques, key, jnp.inf)

    idx = jnp.argsort(key, stable=True)

    # Sort both key and value arrays using the same index
    sorted_key = key[idx]
    sorted_val = jax.tree_util.tree_map(lambda x: x[idx], val)
    return sorted_key[:n], sorted_val[:n], sorted_key[n:], sorted_val[n:]


def pop_full(search_result: SearchResult):
    search_result.priority_queue, min_key, min_val = BGPQ.delete_mins(search_result.priority_queue)
    min_val_cost = min_val.current.cost
    optimal = jnp.less(
        min_val_cost, search_result.cost[min_val.current.index, min_val.current.table_index]
    )
    min_key = jnp.where(optimal, min_key, jnp.inf)
    min_key, min_val, search_result.min_key_buffer, search_result.min_val_buffer = merge_sort_split(
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
