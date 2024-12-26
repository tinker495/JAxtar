from functools import partial

import chex
import jax
import jax.numpy as jnp

from heuristic.heuristic_base import Heuristic
from JAxtar.annotate import ACTION_DTYPE, KEY_DTYPE, SIZE_DTYPE
from JAxtar.bgpq import BGPQ, HashTableIdx_HeapValue
from JAxtar.hash import HashTable, hash_func_builder
from JAxtar.search_base import SearchResult, pop_full
from puzzle.puzzle_base import Puzzle


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

    batch_size = jnp.array(batch_size, dtype=SIZE_DTYPE)
    max_nodes = jnp.array(max_nodes, dtype=SIZE_DTYPE)
    hash_func = hash_func_builder(puzzle.State)
    search_result_build = partial(SearchResult.build, statecls, batch_size, max_nodes)

    parallel_insert = partial(HashTable.parallel_insert, hash_func)
    solved_fn = jax.vmap(puzzle.is_solved, in_axes=(0, None))
    neighbours_fn = jax.vmap(puzzle.get_neighbours, in_axes=(0, 0), out_axes=(1, 1))

    def astar(
        search_result: SearchResult,
        start: Puzzle.State,
        filled: chex.Array,
        target: Puzzle.State,
    ) -> tuple[SearchResult, chex.Array]:
        """
        astar is the implementation of the A* algorithm.
        """

        states = start

        heur_val = heuristic.batched_distance(states, target)
        search_result.hashtable, inserted, _, idx, table_idx = parallel_insert(
            search_result.hashtable, states, filled
        )
        hash_idxs = HashTableIdx_HeapValue(index=idx, table_index=table_idx)

        cost_val = jnp.where(filled, 0, jnp.inf)
        search_result.cost = search_result.cost.at[idx, table_idx].set(
            jnp.where(inserted, cost_val, search_result.cost[idx, table_idx])
        )

        total_cost = (cost_val + heur_val).astype(KEY_DTYPE)
        search_result.priority_queue = BGPQ.insert(
            search_result.priority_queue, total_cost, hash_idxs
        )

        def _cond(search_result: SearchResult):
            heap_size = search_result.priority_queue.size
            hash_size = search_result.hashtable.size
            size_cond1 = heap_size > 0  # queue is not empty
            size_cond2 = hash_size < max_nodes  # hash table is not full
            size_cond = jnp.logical_and(size_cond1, size_cond2)

            min_val = search_result.priority_queue.val_store[0]  # get the minimum value
            states = search_result.hashtable.table[min_val.index, min_val.table_index]
            solved = solved_fn(states, target)
            return jnp.logical_and(size_cond, ~solved.any())

        def _body(search_result: SearchResult):
            search_result, parent_idx, filled = pop_full(search_result)

            cost_val = search_result.cost[parent_idx.index, parent_idx.table_index]
            states = search_result.hashtable.table[parent_idx.index, parent_idx.table_index]

            neighbours, ncost = neighbours_fn(states, filled)
            parent_action = jnp.arange(ncost.shape[0], dtype=ACTION_DTYPE)
            nextcosts = cost_val[jnp.newaxis, :] + ncost  # [n_neighbours, batch_size]

            def _scan(search_result: SearchResult, val):
                neighbour, neighbour_cost, parent_action = val
                neighbour_heur = heuristic.batched_distance(neighbour, target)
                neighbour_key = (cost_weight * neighbour_cost + neighbour_heur).astype(KEY_DTYPE)
                filled = jnp.isfinite(neighbour_cost)  # [n_neighbours, batch_size]

                search_result.hashtable, _, _, idx, table_idx = parallel_insert(
                    search_result.hashtable, neighbour, filled
                )

                optimal = jnp.less(neighbour_cost, search_result.cost[idx, table_idx])
                search_result.cost = search_result.cost.at[idx, table_idx].min(
                    neighbour_cost
                )  # update the minimul cost

                search_result.parent = jax.tree_util.tree_map(
                    lambda insert, parent: parent.at[idx, table_idx].set(
                        jnp.where(optimal, insert, parent[idx, table_idx])
                    ),
                    parent_idx,
                    search_result.parent,
                )

                search_result.parent_action = search_result.parent_action.at[idx, table_idx].set(
                    jnp.where(
                        optimal,
                        parent_action,
                        search_result.parent_action[idx, table_idx],
                    )
                )

                vals = HashTableIdx_HeapValue(index=idx, table_index=table_idx)

                search_result.priority_queue = BGPQ.insert(
                    search_result.priority_queue,
                    neighbour_key,
                    vals,
                    added_size=jnp.sum(filled, dtype=SIZE_DTYPE),
                )
                return search_result, None

            search_result, _ = jax.lax.scan(
                _scan, search_result, (neighbours, nextcosts, parent_action)
            )
            return search_result

        search_result = jax.lax.while_loop(_cond, _body, search_result)
        min_val = search_result.priority_queue.val_store[0]  # get the minimum value
        states = search_result.hashtable.table[min_val.index, min_val.table_index]
        solved = solved_fn(states, target)
        solved_idx = min_val[jnp.argmax(solved)]
        return search_result, solved.any(), solved_idx

    return search_result_build, jax.jit(astar)
