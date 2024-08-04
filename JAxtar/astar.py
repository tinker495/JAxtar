import chex
import jax.numpy as jnp
import jax
from functools import partial
from puzzle.puzzle_base import Puzzle
from JAxtar.bgpq import BGPQ, HashTableIdx_HeapValue
from JAxtar.hash import HashTable



@chex.dataclass
class OpenClosedSet:
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
    heuristic: chex.Array
    closed: chex.Array

    @staticmethod
    def build(statecls: Puzzle.State, HashTablecls: HashTable, batch_size: int, max_nodes: int, seed=0):
        '''
        build is a static method that creates a new instance of OpenClosedSet.
        '''
        hashtable = HashTablecls.build(statecls, seed, max_nodes)
        n_table = hashtable.n_table
        priority_queue = BGPQ.build(max_nodes * n_table, batch_size, HashTableIdx_HeapValue)
        cost = jnp.full((max_nodes, n_table), jnp.inf)
        heuristic = jnp.full((max_nodes, n_table), jnp.inf)
        closed = jnp.zeros((max_nodes, n_table), dtype=jnp.bool)
        return OpenClosedSet(
            hashtable=hashtable,
            priority_queue=priority_queue,
            cost=cost,
            heuristic=heuristic,
            closed=closed
        )
    
    @staticmethod
    def update(
        insert_fn: HashTable.parallel_insert,
        open_closed: "OpenClosedSet",
        state: Puzzle.State,
        idx: int,
        cost: float,
        heuristic: float,
        closed: bool
    ):
        '''
        update updates the OpenClosedSet with the new state.
        '''
        open_closed.hashtable = insert_fn(
            open_closed.hashtable,
            state,
            idx
        )


def astar(
    puzzle: Puzzle,
    start: Puzzle.State,
    target: Puzzle.State,
    heuristic_fn: callable,
    batch_size: int = 1024,
    max_nodes: int = int(1e6)
):
    '''
    astar is the implementation of the A* algorithm.
    '''