from functools import partial

import chex
import jax
import jax.numpy as jnp

from JAxtar.bgpq import BGPQ, HashTableIdx_HeapValue
from JAxtar.hash import HashTable, hash_func_builder
from JAxtar.search_result import SearchResult, pop_full
from puzzle.puzzle_base import Puzzle
from qfunction.q_base import QFunction


def qstar_builder(
    puzzle: Puzzle,
    q_fn: QFunction,
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
    qstar_result_build = partial(SearchResult.build, statecls, batch_size, max_nodes)

    parallel_insert = partial(HashTable.parallel_insert, hash_func)
    solved_fn = jax.vmap(puzzle.is_solved, in_axes=(0, None))
    neighbours_fn = jax.vmap(puzzle.get_neighbours, in_axes=(0, 0), out_axes=(1, 1))

    def qstar(
        qstar_result: SearchResult,
        start: Puzzle.State,
        filled: chex.Array,
        target: Puzzle.State,
    ) -> tuple[SearchResult, chex.Array]:
        """
        astar is the implementation of the A* algorithm.
        """

        states = start

        qstar_result.hashtable, inserted, idx, table_idx = parallel_insert(
            qstar_result.hashtable, states, filled
        )
        hash_idxs = HashTableIdx_HeapValue(index=idx, table_index=table_idx)[:, jnp.newaxis]

        cost_val = jnp.where(filled, 0, jnp.inf)
        qstar_result.cost = qstar_result.cost.at[idx, table_idx].set(
            jnp.where(inserted, cost_val, qstar_result.cost[idx, table_idx])
        )

        total_cost = cost_val  # no heuristic in Q* and first key is no matter
        qstar_result.priority_queue = BGPQ.insert(
            qstar_result.priority_queue, total_cost, hash_idxs
        )

        def _cond(qstar_result: SearchResult):
            heap_size = qstar_result.priority_queue.size
            hash_size = qstar_result.hashtable.size
            size_cond1 = heap_size > 0  # queue is not empty
            size_cond2 = hash_size < max_nodes  # hash table is not full
            size_cond = jnp.logical_and(size_cond1, size_cond2)

            min_val = qstar_result.priority_queue.val_store[0]  # get the minimum value
            states = qstar_result.hashtable.table[min_val.index, min_val.table_index]
            solved = solved_fn(states, target)
            return jnp.logical_and(size_cond, ~solved.any())

        def _body(qstar_result: SearchResult):
            qstar_result, min_val, filled = pop_full(qstar_result)
            min_idx, min_table_idx = min_val.index, min_val.table_index
            parent_idx = jnp.stack((min_idx, min_table_idx), axis=-1)

            cost_val = qstar_result.cost[min_idx, min_table_idx]
            states = qstar_result.hashtable.table[min_idx, min_table_idx]

            qstar_result.not_closed = qstar_result.not_closed.at[min_idx, min_table_idx].min(
                ~filled
            )  # or operation with closed

            q_vals = q_fn.batched_q_value(
                states, target
            ).transpose()  # [batch_size, n_neighbours] -> [n_neighbours, batch_size]
            neighbours, ncost = neighbours_fn(states, filled)
            parent_action = jnp.tile(
                jnp.arange(ncost.shape[0], dtype=jnp.uint32)[:, jnp.newaxis], (1, ncost.shape[1])
            )
            nextcosts = cost_val[jnp.newaxis, :] + ncost  # [n_neighbours, batch_size]
            neighbour_key = cost_weight * nextcosts + q_vals
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
            qstar_result.hashtable, _, idxs, table_idxs = parallel_insert(
                qstar_result.hashtable, flatten_neighbours, filleds.reshape((flatten_size,))
            )

            flatten_nextcosts = nextcosts.reshape((flatten_size,))
            optimals = jnp.less(flatten_nextcosts, qstar_result.cost[idxs, table_idxs])
            qstar_result.cost = qstar_result.cost.at[idxs, table_idxs].min(
                flatten_nextcosts
            )  # update the minimul cost

            flatten_neighbours_parent_idx = neighbours_parent_idx.reshape((flatten_size, 2))
            flatten_parent_action = parent_action.reshape((flatten_size,))
            qstar_result.parent = qstar_result.parent.at[idxs, table_idxs].set(
                jnp.where(
                    optimals[:, jnp.newaxis],
                    flatten_neighbours_parent_idx,
                    qstar_result.parent[idxs, table_idxs],
                )
            )
            qstar_result.parent_action = qstar_result.parent_action.at[idxs, table_idxs].set(
                jnp.where(
                    optimals,
                    flatten_parent_action,
                    qstar_result.parent_action[idxs, table_idxs],
                )
            )

            idxs = idxs.reshape(unflatten_size)
            table_idxs = table_idxs.reshape(unflatten_size)
            optimals = optimals.reshape(unflatten_size)

            def _scan(qstar_result: SearchResult, val):
                neighbour_key, idx, table_idx, optimal = val

                vals = HashTableIdx_HeapValue(index=idx, table_index=table_idx)[:, jnp.newaxis]
                not_closed_update = qstar_result.not_closed[idx, table_idx] | optimal
                qstar_result.not_closed = qstar_result.not_closed.at[idx, table_idx].set(
                    not_closed_update
                )
                neighbour_key = jnp.where(not_closed_update, neighbour_key, jnp.inf)

                qstar_result.priority_queue = BGPQ.insert(
                    qstar_result.priority_queue,
                    neighbour_key,
                    vals,
                    added_size=jnp.sum(optimal, dtype=jnp.uint32),
                )
                return qstar_result, None

            qstar_result, _ = jax.lax.scan(
                _scan, qstar_result, (neighbour_key, idxs, table_idxs, optimals)
            )
            return qstar_result

        qstar_result = jax.lax.while_loop(_cond, _body, qstar_result)
        min_val = qstar_result.priority_queue.val_store[0]  # get the minimum value
        states = qstar_result.hashtable.table[min_val.index, min_val.table_index]
        solved = solved_fn(states, target)
        solved_idx = min_val[jnp.argmax(solved)]
        return qstar_result, solved.any(), solved_idx

    return qstar_result_build, jax.jit(qstar)
