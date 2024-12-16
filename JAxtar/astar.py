from functools import partial

import chex
import jax
import jax.numpy as jnp

from heuristic.heuristic_base import Heuristic
from JAxtar.bgpq import BGPQ, HashTableIdx_HeapValue, HeapValue
from JAxtar.hash import HashTable, hash_func_builder
from puzzle.puzzle_base import Puzzle


@chex.dataclass
class AstarResult:
    """
    OpenClosedSet is a dataclass that contains the data structures used in the A* algorithm.

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
        return AstarResult(
            hashtable=hashtable,
            priority_queue=priority_queue,
            min_key_buffer=min_key_buffer,
            min_val_buffer=min_val_buffer,
            cost=cost,
            not_closed=not_closed,
            parent=parent,
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


def astar_builder(
    puzzle: Puzzle,
    heuristic: Heuristic,
    batch_size: int = 1024,
    max_nodes: int = int(1e6),
    cost_weight: float = 1.0 - 1e-6,
):
    """
    astar_builder is a function that returns a partial function of astar.

    Args:
    - puzzle: Puzzle instance that contains the puzzle.
    - heuristic_fn: heuristic function that returns the heuristic value of the states.
    - batch_size: batch size of the states.
    - max_nodes: maximum number of nodes that can be stored in the HashTable.
    - astar_weight: weight of the cost function in the A* algorithm.
    - efficient_heuristic: if True, the heuristic value of the states is stored in the HashTable.
                        This is useful when the heuristic function is expensive to compute.
                        ex) neural heuristic function.
                        This option is slower than the normal heuristic function
                        because of the overhead of the HashTable.
    """

    statecls = puzzle.State

    batch_size = jnp.array(batch_size, dtype=jnp.int32)
    max_nodes = jnp.array(max_nodes, dtype=jnp.int32)
    hash_func = hash_func_builder(puzzle.State)
    astar_result_build = partial(AstarResult.build, statecls, batch_size, max_nodes)

    parallel_insert = partial(HashTable.parallel_insert, hash_func)
    solved_fn = jax.vmap(puzzle.is_solved, in_axes=(0, None))
    neighbours_fn = jax.vmap(puzzle.get_neighbours, in_axes=(0, 0), out_axes=(1, 1))

    def astar(
        astar_result: AstarResult,
        start: Puzzle.State,
        filled: chex.Array,
        target: Puzzle.State,
    ) -> tuple[AstarResult, chex.Array]:
        """
        astar is the implementation of the A* algorithm.
        """

        states = start

        heur_val = heuristic.batched_distance(states, target)
        astar_result.hashtable, inserted, idx, table_idx = parallel_insert(
            astar_result.hashtable, states, filled
        )
        hash_idxs = HashTableIdx_HeapValue(index=idx, table_index=table_idx)[:, jnp.newaxis]

        cost_val = jnp.where(filled, 0, jnp.inf)
        astar_result.cost = astar_result.cost.at[idx, table_idx].set(
            jnp.where(inserted, cost_val, astar_result.cost[idx, table_idx])
        )

        total_cost = cost_val + heur_val
        astar_result.priority_queue = BGPQ.insert(
            astar_result.priority_queue, total_cost, hash_idxs
        )

        def _cond(astar_result: AstarResult):
            heap_size = astar_result.priority_queue.size
            hash_size = astar_result.hashtable.size
            size_cond1 = heap_size > 0  # queue is not empty
            size_cond2 = hash_size < max_nodes  # hash table is not full
            size_cond = jnp.logical_and(size_cond1, size_cond2)

            min_val = astar_result.priority_queue.val_store[0]  # get the minimum value
            states = astar_result.hashtable.table[min_val.index, min_val.table_index]
            solved = solved_fn(states, target)
            return jnp.logical_and(size_cond, ~solved.any())

        def _body(astar_result: AstarResult):
            astar_result, min_val, filled = pop_full(astar_result)
            min_idx, min_table_idx = min_val.index, min_val.table_index
            parent_idx = jnp.stack((min_idx, min_table_idx), axis=-1)

            cost_val = astar_result.cost[min_idx, min_table_idx]
            states = astar_result.hashtable.table[min_idx, min_table_idx]

            astar_result.not_closed = astar_result.not_closed.at[min_idx, min_table_idx].min(
                ~filled
            )  # or operation with closed

            neighbours, ncost = neighbours_fn(states, filled)
            nextcosts = cost_val[jnp.newaxis, :] + ncost  # [n_neighbours, batch_size]
            filleds = jnp.isfinite(nextcosts)  # [n_neighbours, batch_size]
            neighbours_parent_idx = jnp.broadcast_to(
                parent_idx, (filleds.shape[0], filleds.shape[1], 2)
            )

            # insert neighbours into hashtable at once
            unflatten_size = filleds.shape
            flatten_size = unflatten_size[0] * unflatten_size[1]

            flatten_neighbours = jax.tree_util.tree_map(
                lambda x: x.reshape((flatten_size, *x.shape[2:])), neighbours
            )
            astar_result.hashtable, updated, idxs, table_idxs = parallel_insert(
                astar_result.hashtable, flatten_neighbours, filleds.reshape((flatten_size,))
            )

            flatten_nextcosts = nextcosts.reshape((flatten_size,))
            optimals = jnp.less(flatten_nextcosts, astar_result.cost[idxs, table_idxs])
            astar_result.cost = astar_result.cost.at[idxs, table_idxs].min(
                flatten_nextcosts
            )  # update the minimul cost

            flatten_neighbours_parent_idx = neighbours_parent_idx.reshape((flatten_size, 2))
            astar_result.parent = astar_result.parent.at[idxs, table_idxs].set(
                jnp.where(
                    optimals[:, jnp.newaxis],
                    flatten_neighbours_parent_idx,
                    astar_result.parent[idxs, table_idxs],
                )
            )

            idxs = idxs.reshape(unflatten_size)
            table_idxs = table_idxs.reshape(unflatten_size)
            optimals = optimals.reshape(unflatten_size)

            def _scan(astar_result: AstarResult, val):
                neighbour, neighbour_cost, idx, table_idx, optimal = val
                neighbour_heur = heuristic.batched_distance(neighbour, target)
                neighbour_key = cost_weight * neighbour_cost + neighbour_heur

                vals = HashTableIdx_HeapValue(index=idx, table_index=table_idx)[:, jnp.newaxis]
                not_closed_update = astar_result.not_closed[idx, table_idx] | optimal
                astar_result.not_closed = astar_result.not_closed.at[idx, table_idx].set(
                    not_closed_update
                )
                neighbour_key = jnp.where(not_closed_update, neighbour_key, jnp.inf)

                astar_result.priority_queue = BGPQ.insert(
                    astar_result.priority_queue,
                    neighbour_key,
                    vals,
                    added_size=jnp.sum(optimal, dtype=jnp.uint32),
                )
                return astar_result, None

            astar_result, _ = jax.lax.scan(
                _scan, astar_result, (neighbours, nextcosts, idxs, table_idxs, optimals)
            )
            return astar_result

        astar_result = jax.lax.while_loop(_cond, _body, astar_result)
        min_val = astar_result.priority_queue.val_store[0]  # get the minimum value
        states = astar_result.hashtable.table[min_val.index, min_val.table_index]
        solved = solved_fn(states, target)
        solved_idx = min_val[jnp.argmax(solved)]
        return astar_result, solved.any(), solved_idx

    return astar_result_build, jax.jit(astar)


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


def pop_full(astar_result: AstarResult):
    astar_result.priority_queue, min_key, min_val = BGPQ.delete_mins(astar_result.priority_queue)
    min_idx, min_table_idx = min_val.index, min_val.table_index
    min_key = jnp.where(astar_result.not_closed[min_idx, min_table_idx], min_key, jnp.inf)
    min_key, min_val, astar_result.min_key_buffer, astar_result.min_val_buffer = merge_sort_split(
        min_key, min_val, astar_result.min_key_buffer, astar_result.min_val_buffer
    )
    filled = jnp.isfinite(min_key)

    def _cond(val):
        astar_result, _, _, filled = val
        return jnp.logical_and(astar_result.priority_queue.size > 0, ~filled.all())

    def _body(val):
        astar_result, min_key, min_val, filled = val
        astar_result.priority_queue, min_key_buffer, min_val_buffer = BGPQ.delete_mins(
            astar_result.priority_queue
        )
        min_key_buffer = jnp.where(
            astar_result.not_closed[min_val_buffer.index, min_val_buffer.table_index],
            min_key_buffer,
            jnp.inf,
        )
        (
            min_key,
            min_val,
            astar_result.min_key_buffer,
            astar_result.min_val_buffer,
        ) = merge_sort_split(min_key, min_val, min_key_buffer, min_val_buffer)
        filled = jnp.isfinite(min_key)
        return astar_result, min_key, min_val, filled

    astar_result, min_key, min_val, filled = jax.lax.while_loop(
        _cond, _body, (astar_result, min_key, min_val, filled)
    )
    return astar_result, min_val, filled
