import chex
import jax
import jax.numpy as jnp

from JAxtar.bgpq import BGPQ, HashTableIdx_HeapValue, HeapValue
from JAxtar.hash import HashTable
from puzzle.puzzle_base import Puzzle


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
    min_val_buffer: HashTableIdx_HeapValue
    cost: chex.Array
    not_closed: chex.Array
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
        priority_queue = BGPQ.build(max_nodes, batch_size, HashTableIdx_HeapValue)
        min_key_buffer = jnp.full((batch_size,), jnp.inf)
        min_val_buffer = HashTableIdx_HeapValue(
            index=jnp.zeros((batch_size,), dtype=jnp.uint32),
            table_index=jnp.zeros((batch_size,), dtype=jnp.uint32),
        )
        cost = jnp.full((size_table, n_table), jnp.inf)
        not_closed = jnp.ones((size_table, n_table), dtype=jnp.bool)
        parent = jnp.full((size_table, n_table, 2), -1, dtype=jnp.uint32)
        parent_action = jnp.full((size_table, n_table), -1, dtype=jnp.uint32)
        return SearchResult(
            hashtable=hashtable,
            priority_queue=priority_queue,
            min_key_buffer=min_key_buffer,
            min_val_buffer=min_val_buffer,
            cost=cost,
            not_closed=not_closed,
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
    idx = jnp.argsort(key, stable=True)

    # Sort both key and value arrays using the same index
    sorted_key = key[idx]
    sorted_val = jax.tree_util.tree_map(lambda x: x[idx], val)
    return sorted_key[:n], sorted_val[:n], sorted_key[n:], sorted_val[n:]


def pop_full(search_result: SearchResult):
    search_result.priority_queue, min_key, min_val = BGPQ.delete_mins(search_result.priority_queue)
    min_idx, min_table_idx = min_val.index, min_val.table_index
    min_key = jnp.where(search_result.not_closed[min_idx, min_table_idx], min_key, jnp.inf)
    min_key, min_val, search_result.min_key_buffer, search_result.min_val_buffer = merge_sort_split(
        min_key, min_val, search_result.min_key_buffer, search_result.min_val_buffer
    )
    filled = jnp.isfinite(min_key)

    def _cond(val):
        search_result, _, _, filled = val
        return jnp.logical_and(search_result.priority_queue.size > 0, ~filled.all())

    def _body(val):
        search_result, min_key, min_val, filled = val
        search_result.priority_queue, min_key_buffer, min_val_buffer = BGPQ.delete_mins(
            search_result.priority_queue
        )
        min_key_buffer = jnp.where(
            search_result.not_closed[min_val_buffer.index, min_val_buffer.table_index],
            min_key_buffer,
            jnp.inf,
        )
        (
            min_key,
            min_val,
            search_result.min_key_buffer,
            search_result.min_val_buffer,
        ) = merge_sort_split(min_key, min_val, min_key_buffer, min_val_buffer)
        filled = jnp.isfinite(min_key)
        return search_result, min_key, min_val, filled

    search_result, min_key, min_val, filled = jax.lax.while_loop(
        _cond, _body, (search_result, min_key, min_val, filled)
    )
    return search_result, min_val, filled
