import chex
import jax.numpy as jnp
import jax
from functools import partial
from puzzle.puzzle_base import Puzzle
from JAxtar.bgpq import BGPQ, HashTableIdx_HeapValue, HeapValue
from JAxtar.hash import hash_func_builder, HashTable


@chex.dataclass
class QstarResult:
    '''
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
    - parant: a 2D array that contains the index of the parent node.
    '''

    hashtable: HashTable
    priority_queue: BGPQ
    min_key_buffer: chex.Array
    min_val_buffer: HashTableIdx_HeapValue
    cost: chex.Array
    not_closed: chex.Array
    parant: chex.Array

    @staticmethod
    def build(statecls: Puzzle.State, batch_size: int, max_nodes: int, seed=0, n_table=2):
        '''
        build is a static method that creates a new instance of AstarResult.
        '''
        hashtable = HashTable.build(statecls, seed, max_nodes, n_table=n_table)
        size_table = hashtable.capacity
        n_table = hashtable.n_table
        priority_queue = BGPQ.build(max_nodes, batch_size, HashTableIdx_HeapValue)
        min_key_buffer = jnp.full((batch_size, ), jnp.inf)
        min_val_buffer = HashTableIdx_HeapValue(index=jnp.zeros((batch_size, ), dtype=jnp.uint32), table_index=jnp.zeros((batch_size, ), dtype=jnp.uint32))
        cost = jnp.full((size_table, n_table), jnp.inf)
        not_closed = jnp.ones((size_table, n_table), dtype=jnp.bool)
        parant = jnp.full((size_table, n_table, 2), -1, dtype=jnp.uint32)
        return QstarResult(
            hashtable=hashtable,
            priority_queue=priority_queue,
            min_key_buffer=min_key_buffer,
            min_val_buffer=min_val_buffer,
            cost=cost,
            not_closed=not_closed,
            parant=parant
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

def qstar_builder(puzzle: Puzzle, q_fn: callable, batch_size: int = 1024, max_nodes: int = int(1e6), astar_weight : float = 1.0 - 1e-6):
    '''
    astar_builder is a function that returns a partial function of astar.

    Args:
    - puzzle: Puzzle instance that contains the puzzle.
    - heuristic_fn: heuristic function that returns the heuristic value of the states.
    - batch_size: batch size of the states.
    - max_nodes: maximum number of nodes that can be stored in the HashTable.
    - astar_weight: weight of the cost function in the A* algorithm.
    - efficient_heuristic: if True, the heuristic value of the states is stored in the HashTable.
                        This is useful when the heuristic function is expensive to compute. ex) neural heuristic function.
                        This option is slower than the normal heuristic function because of the overhead of the HashTable.
    '''
    
    statecls = puzzle.State

    batch_size = jnp.array(batch_size, dtype=jnp.int32)
    max_nodes = jnp.array(max_nodes, dtype=jnp.int32)
    hash_func = hash_func_builder(puzzle.State)
    qstar_result_build = partial(QstarResult.build, statecls, batch_size, max_nodes)
    
    q_fn = jax.vmap(q_fn, in_axes=(0, None))

    parallel_insert = partial(HashTable.parallel_insert, hash_func)
    solved_fn = jax.vmap(puzzle.is_solved, in_axes=(0, None))
    neighbours_fn = jax.vmap(puzzle.get_neighbours, in_axes=(0,0), out_axes=(1,1))

    def qstar(
        qstar_result: QstarResult,
        start: Puzzle.State,
        filled: chex.Array,
        target: Puzzle.State,
    ) -> tuple[QstarResult, chex.Array]:
        '''
        astar is the implementation of the A* algorithm.
        '''

        states = start

        q_val = q_fn(states, target)
        qstar_result.hashtable, inserted, idx, table_idx = parallel_insert(qstar_result.hashtable, states, filled)
        hash_idxs = HashTableIdx_HeapValue(index=idx, table_index=table_idx)[:, jnp.newaxis]

        cost_val = jnp.where(filled, 0, jnp.inf)
        qstar_result.cost = qstar_result.cost.at[idx, table_idx].set(jnp.where(inserted, cost_val, qstar_result.cost[idx, table_idx]))
        
        total_cost = cost_val + q_val
        qstar_result.priority_queue = BGPQ.insert(qstar_result.priority_queue, total_cost, hash_idxs)

        def _cond(qstar_result: QstarResult):
            heap_size = qstar_result.priority_queue.size
            hash_size = qstar_result.hashtable.size
            size_cond1 = heap_size > 0 # queue is not empty
            size_cond2 = hash_size < max_nodes # hash table is not full
            size_cond = jnp.logical_and(size_cond1, size_cond2)

            min_val = qstar_result.priority_queue.val_store[0] # get the minimum value
            states = qstar_result.hashtable.table[min_val.index, min_val.table_index]
            solved = solved_fn(states, target)
            return jnp.logical_and(size_cond, ~solved.any())
        
        def _body(qstar_result: QstarResult):
            qstar_result, min_val, filled = pop_full(qstar_result)
            min_idx, min_table_idx = min_val.index, min_val.table_index
            parant_idx = jnp.stack((min_idx, min_table_idx), axis=-1)

            cost_val = qstar_result.cost[min_idx, min_table_idx]
            states = qstar_result.hashtable.table[min_idx, min_table_idx]

            qstar_result.not_closed = qstar_result.not_closed.at[min_idx, min_table_idx].min(~filled) # or operation with closed

            neighbours, ncost = neighbours_fn(states, filled)
            nextcosts = cost_val[jnp.newaxis, :] + ncost # [n_neighbours, batch_size]
            filleds = jnp.isfinite(nextcosts) # [n_neighbours, batch_size]
            neighbours_parant_idx = jnp.broadcast_to(parant_idx, (filleds.shape[0], filleds.shape[1], 2))

            #insert neighbours into hashtable at once
            unflatten_size = filleds.shape
            flatten_size = unflatten_size[0] * unflatten_size[1]
            
            flatten_neighbours = jax.tree_util.tree_map(lambda x: x.reshape((flatten_size, *x.shape[2:])), neighbours)
            qstar_result.hashtable, _, idxs, table_idxs = parallel_insert(qstar_result.hashtable, flatten_neighbours, filleds.reshape((flatten_size,)))
            
            flatten_nextcosts = nextcosts.reshape((flatten_size,))
            optimals = jnp.less(flatten_nextcosts, qstar_result.cost[idxs, table_idxs])
            qstar_result.cost = qstar_result.cost.at[idxs, table_idxs].min(flatten_nextcosts) # update the minimul cost

            flatten_neighbours_parant_idx = neighbours_parant_idx.reshape((flatten_size, 2))
            qstar_result.parant = qstar_result.parant.at[idxs, table_idxs].set(jnp.where(optimals[:,jnp.newaxis], flatten_neighbours_parant_idx, qstar_result.parant[idxs, table_idxs]))

            idxs = idxs.reshape(unflatten_size)
            table_idxs = table_idxs.reshape(unflatten_size)
            optimals = optimals.reshape(unflatten_size)

            def _scan(qstar_result : QstarResult, val):
                neighbour, neighbour_cost, idx, table_idx, optimal = val
                neighbour_heur = q_fn(neighbour, target)
                neighbour_key = astar_weight * neighbour_cost + neighbour_heur

                vals = HashTableIdx_HeapValue(index=idx, table_index=table_idx)[:, jnp.newaxis]
                not_closed_update =  qstar_result.not_closed[idx, table_idx] | optimal
                qstar_result.not_closed = qstar_result.not_closed.at[idx, table_idx].set(not_closed_update)
                neighbour_key = jnp.where(not_closed_update, neighbour_key, jnp.inf)

                qstar_result.priority_queue = BGPQ.insert(qstar_result.priority_queue, neighbour_key, vals, added_size=jnp.sum(optimal, dtype=jnp.uint32))
                return qstar_result, None

            qstar_result, _ = jax.lax.scan(_scan, qstar_result, (neighbours, nextcosts, idxs, table_idxs, optimals))
            return qstar_result
        
        qstar_result = jax.lax.while_loop(_cond, _body, qstar_result)
        min_val = qstar_result.priority_queue.val_store[0] # get the minimum value
        states = qstar_result.hashtable.table[min_val.index, min_val.table_index]
        solved = solved_fn(states, target)
        solved_idx = min_val[jnp.argmax(solved)]
        return qstar_result, solved.any(), solved_idx

    return qstar_result_build, jax.jit(qstar)

def merge_sort_split(ak: chex.Array, av: HeapValue, bk: chex.Array, bv: HeapValue) -> tuple[chex.Array, HeapValue, chex.Array, HeapValue]:
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
    n = ak.shape[-1] # size of group
    key = jnp.concatenate([ak, bk])
    val = jax.tree_util.tree_map(lambda a, b: jnp.concatenate([a, b]), av, bv)
    idx = jnp.argsort(key, stable=True)

    # Sort both key and value arrays using the same index
    sorted_key = key[idx]
    sorted_val = jax.tree_util.tree_map(lambda x: x[idx], val)
    return sorted_key[:n], sorted_val[:n], sorted_key[n:], sorted_val[n:]

def pop_full(qstar_result: QstarResult):
    qstar_result.priority_queue, min_key, min_val = BGPQ.delete_mins(qstar_result.priority_queue)
    min_idx, min_table_idx = min_val.index, min_val.table_index
    min_key = jnp.where(qstar_result.not_closed[min_idx, min_table_idx], min_key, jnp.inf)
    min_key, min_val, qstar_result.min_key_buffer, qstar_result.min_val_buffer = merge_sort_split(min_key, min_val, qstar_result.min_key_buffer, qstar_result.min_val_buffer)
    filled = jnp.isfinite(min_key)

    def _cond(val):
        qstar_result, _, _, filled = val
        return jnp.logical_and(qstar_result.priority_queue.size > 0, ~filled.all())
    
    def _body(val):
        qstar_result, min_key, min_val, filled = val
        qstar_result.priority_queue, min_key_buffer, min_val_buffer = BGPQ.delete_mins(qstar_result.priority_queue)
        min_key_buffer = jnp.where(qstar_result.not_closed[min_val_buffer.index, min_val_buffer.table_index], min_key_buffer, jnp.inf)
        min_key, min_val, qstar_result.min_key_buffer, qstar_result.min_val_buffer = merge_sort_split(min_key, min_val, min_key_buffer, min_val_buffer)
        filled = jnp.isfinite(min_key)
        return qstar_result, min_key, min_val, filled
    qstar_result, min_key, min_val, filled = jax.lax.while_loop(_cond, _body, (qstar_result, min_key, min_val, filled))
    return qstar_result, min_val, filled
