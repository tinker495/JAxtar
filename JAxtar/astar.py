import chex
import jax.numpy as jnp
import jax
from functools import partial
from puzzle.puzzle_base import Puzzle
from JAxtar.bgpq import BGPQ, HashTableIdx_HeapValue
from JAxtar.hash import hash_func_builder, HashTable



@chex.dataclass
class AstarResult:
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
    - heuristic: estimated cost from the current node to the goal node. 
                 this is not updated during the search, it is a constant value.
    - closed: a boolean array that indicates whether the state is in the closed set or not.
    '''

    hashtable: HashTable
    priority_queue: BGPQ
    cost: chex.Array
    closed: chex.Array
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
        cost = jnp.full((size_table, n_table), jnp.inf)
        closed = jnp.zeros((size_table, n_table), dtype=jnp.bool)
        parant = jnp.full((size_table, n_table, 2), -1, dtype=jnp.int32)
        return AstarResult(
            hashtable=hashtable,
            priority_queue=priority_queue,
            cost=cost,
            closed=closed,
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

def astar_builder(puzzle: Puzzle, heuristic_fn: callable, batch_size: int = 1024, max_nodes: int = int(1e6), astar_weight : float = 1.0 - 1e-6):
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
    astar_result = AstarResult.build(statecls, batch_size, max_nodes)
    
    heuristic = jax.jit(jax.vmap(heuristic_fn, in_axes=(0, None)))

    parallel_insert = jax.jit(partial(HashTable.parallel_insert, hash_func))
    solved_fn = jax.jit(jax.vmap(puzzle.is_solved, in_axes=(0, None)))
    neighbours_fn = jax.jit(jax.vmap(puzzle.get_neighbours, in_axes=(0,0)))
    delete_fn = jax.jit(BGPQ.delete_mins)
    insert_fn = jax.jit(BGPQ.insert)

    def astar(
        astar_result: AstarResult,
        start: Puzzle.State,
        filled: chex.Array,
        target: Puzzle.State,
    ) -> tuple[AstarResult, chex.Array]:
        '''
        astar is the implementation of the A* algorithm.
        '''
        
        states = start

        heur_val = heuristic(states, target)
        astar_result.hashtable, inserted, idx, table_idx = parallel_insert(astar_result.hashtable, states, filled)
        hash_idxs = HashTableIdx_HeapValue(index=idx, table_index=table_idx)[:, jnp.newaxis]

        cost_val = jnp.where(filled, 0, jnp.inf)
        astar_result.cost = astar_result.cost.at[idx, table_idx].set(jnp.where(inserted, cost_val, astar_result.cost[idx, table_idx]))
        
        total_cost = cost_val + heur_val
        astar_result.priority_queue = BGPQ.insert(astar_result.priority_queue, total_cost, hash_idxs)

        def _cond(astar_result: AstarResult):
            heap_size = astar_result.priority_queue.size
            hash_size = astar_result.hashtable.size
            size_cond1 = heap_size > 0
            size_cond2 = heap_size < max_nodes
            size_cond3 = hash_size < max_nodes
            size_cond = jnp.logical_and(jnp.logical_and(size_cond1, size_cond2), size_cond3)

            min_val = astar_result.priority_queue.val_store[0] # get the minimum value
            states = astar_result.hashtable.table[min_val.index, min_val.table_index]
            solved = solved_fn(states, target)
            return jnp.logical_and(size_cond, ~solved.any())
        
        def _body(astar_result: AstarResult):
            astar_result.priority_queue, min_key, min_val = delete_fn(astar_result.priority_queue)
            min_idx, min_table_idx = min_val.index, min_val.table_index
            parant_idx = jnp.stack((min_idx, min_table_idx), axis=-1).astype(jnp.int32)

            cost_val, closed_val = astar_result.cost[min_idx, min_table_idx], astar_result.closed[min_idx, min_table_idx]
            states = astar_result.hashtable.table[min_idx, min_table_idx]

            filled = jnp.logical_and(jnp.isfinite(min_key),~closed_val)

            def _filled(astar_result: AstarResult):
                astar_result.closed = astar_result.closed.at[min_idx, min_table_idx].set(jnp.where(filled, True, astar_result.closed[min_idx, min_table_idx]))

                neighbours, ncost = neighbours_fn(states, filled)
                nextcosts = cost_val[:, jnp.newaxis] + ncost
                filleds = jnp.isfinite(nextcosts)

                nextheur = jax.vmap(heuristic, in_axes=(0, None))(neighbours, target)
                nextkeys = astar_weight * nextcosts + nextheur

                def _scan(astar_result : AstarResult, val):
                    neighbour, neighbour_key, neighbour_cost, neighbour_filled = val

                    astar_result.hashtable, inserted, idx, table_idx = parallel_insert(astar_result.hashtable, neighbour, neighbour_filled)
                    vals = HashTableIdx_HeapValue(index=idx, table_index=table_idx)[:, jnp.newaxis]
                    more_optimal = (neighbour_cost < astar_result.cost[idx, table_idx])
                    astar_result.cost = astar_result.cost.at[idx, table_idx].set(jnp.minimum(neighbour_cost, astar_result.cost[idx, table_idx]))
                    astar_result.parant = astar_result.parant.at[idx, table_idx].set(jnp.where(more_optimal[:,jnp.newaxis], parant_idx, astar_result.parant[idx, table_idx]))
                    astar_result.closed = astar_result.closed.at[idx, table_idx].set(jnp.logical_and(astar_result.closed[idx, table_idx], ~more_optimal))
                    neighbour_key = jnp.where(astar_result.closed[idx, table_idx], jnp.inf, neighbour_key)

                    astar_result.priority_queue = insert_fn(astar_result.priority_queue, neighbour_key, vals)
                    return astar_result, None
                
                neighbours = jax.tree_util.tree_map(lambda x: jnp.moveaxis(x, 0, 1), neighbours)
                nextkeys = jnp.moveaxis(nextkeys, 0, 1)
                nextcosts = jnp.moveaxis(nextcosts, 0, 1)
                filleds = jnp.moveaxis(filleds, 0, 1)
                astar_result, _ = jax.lax.scan(_scan, astar_result, (neighbours, nextkeys, nextcosts, filleds))

                return astar_result
            
            astar_result = jax.lax.cond(filled.any(), _filled, lambda x: x, astar_result)

            return astar_result
        
        astar_result = jax.lax.while_loop(_cond, _body, astar_result)
        min_val = astar_result.priority_queue.val_store[0] # get the minimum value
        states = astar_result.hashtable.table[min_val.index, min_val.table_index]
        solved = solved_fn(states, target)
        solved_idx = min_val[jnp.argmax(solved)]
        return astar_result, solved.any(), solved_idx

    return jax.jit(partial(astar, astar_result))