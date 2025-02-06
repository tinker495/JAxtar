import time
from functools import partial

import chex
import jax
import jax.numpy as jnp

from heuristic.heuristic_base import Heuristic
from JAxtar.annotate import ACTION_DTYPE, KEY_DTYPE, SIZE_DTYPE
from JAxtar.hash import HashTable, hash_func_builder
from JAxtar.search_base import HashTableIdx_HeapValue, SearchResult
from JAxtar.util import (
    flatten_array,
    flatten_tree,
    set_array_as_condition,
    set_tree_as_condition,
    unflatten_array,
)
from puzzle.puzzle_base import Puzzle


def astar_builder(
    puzzle: Puzzle,
    heuristic: Heuristic,
    batch_size: int = 1024,
    max_nodes: int = int(1e6),
    cost_weight: float = 1.0 - 1e-6,
    show_compile_time: bool = False,
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

    _batch_size = jnp.array(batch_size, dtype=SIZE_DTYPE)
    _max_nodes = jnp.array(max_nodes, dtype=SIZE_DTYPE)
    hash_func = hash_func_builder(statecls)
    search_result_build = partial(SearchResult.build, statecls, _batch_size, _max_nodes)

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
        (
            search_result.hashtable,
            inserted,
            _,
            idx,
            table_idx,
        ) = search_result.hashtable.parallel_insert(hash_func, states, filled)
        hash_idxs = HashTableIdx_HeapValue(index=idx, table_index=table_idx)

        cost_val = jnp.where(filled, 0, jnp.inf)
        search_result.cost = set_array_as_condition(
            search_result.cost,
            inserted,
            cost_val,
            idx,
            table_idx,
        )

        total_cost = (cost_val + heur_val).astype(KEY_DTYPE)
        search_result.priority_queue = search_result.priority_queue.insert(total_cost, hash_idxs)

        def _cond(input: tuple[SearchResult, HashTableIdx_HeapValue, chex.Array]):
            search_result, parent, filled = input
            hash_size = search_result.hashtable.size
            size_cond1 = filled.any()  # queue is not empty
            size_cond2 = hash_size < max_nodes  # hash table is not full
            size_cond = jnp.logical_and(size_cond1, size_cond2)

            states = search_result.hashtable.table[parent.index, parent.table_index]
            solved = solved_fn(states, target)
            return jnp.logical_and(size_cond, ~solved.any())

        def _body(input: tuple[SearchResult, HashTableIdx_HeapValue, chex.Array]):
            search_result, parent_idx, filled = input

            cost_val = search_result.cost[parent_idx.index, parent_idx.table_index]
            states = search_result.hashtable.table[parent_idx.index, parent_idx.table_index]

            neighbours, ncost = neighbours_fn(states, filled)
            parent_action = jnp.arange(ncost.shape[0], dtype=ACTION_DTYPE)
            nextcosts = cost_val[jnp.newaxis, :] + ncost  # [n_neighbours, batch_size]
            filleds = jnp.isfinite(nextcosts)  # [n_neighbours, batch_size]

            (
                search_result.hashtable,
                _,
                _,
                idxs,
                table_idxs,
            ) = search_result.hashtable.parallel_insert(
                hash_func, flatten_tree(neighbours, 2), flatten_array(filleds, 2)
            )

            idxs = unflatten_array(idxs, filleds.shape)
            table_idxs = unflatten_array(table_idxs, filleds.shape)

            def _scan(search_result: SearchResult, val):
                neighbour, neighbour_cost, filled, parent_action, idx, table_idx = val
                neighbour_heur = heuristic.batched_distance(neighbour, target)
                neighbour_key = (cost_weight * neighbour_cost + neighbour_heur).astype(KEY_DTYPE)

                optimal = jnp.less(neighbour_cost, search_result.cost[idx, table_idx])
                search_result.cost = search_result.cost.at[idx, table_idx].min(
                    neighbour_cost
                )  # update the minimul cost

                search_result.parent = set_tree_as_condition(
                    search_result.parent,
                    optimal,
                    parent_idx,
                    idx,
                    table_idx,
                )

                search_result.parent_action = set_array_as_condition(
                    search_result.parent_action,
                    optimal,
                    parent_action,
                    idx,
                    table_idx,
                )

                vals = HashTableIdx_HeapValue(index=idx, table_index=table_idx)

                search_result.priority_queue = search_result.priority_queue.insert(
                    neighbour_key,
                    vals,
                    added_size=jnp.sum(filled, dtype=SIZE_DTYPE),
                )
                return search_result, None

            search_result, _ = jax.lax.scan(
                _scan,
                search_result,
                (neighbours, nextcosts, filleds, parent_action, idxs, table_idxs),
            )
            search_result, parent, filled = search_result.pop_full()
            return search_result, parent, filled

        search_result, parent, filled = search_result.pop_full()
        (search_result, idxes, filled) = jax.lax.while_loop(
            _cond, _body, (search_result, parent, filled)
        )
        states = search_result.hashtable.table[idxes.index, idxes.table_index]
        solved = solved_fn(states, target)
        search_result.solved = solved.any()
        search_result.solved_idx = idxes[jnp.argmax(solved)]
        return search_result

    astar_fn = jax.jit(astar)
    inital_search_result = search_result_build()
    empty_target = puzzle.State.default()
    empty_states = puzzle.State.default()[jnp.newaxis, ...]

    empty_states, filled = HashTable.make_batched(puzzle.State, empty_states, batch_size)
    if show_compile_time:
        print("initializing jit")
        start = time.time()

    # Pass empty states and target to JIT-compile the function with simple data.
    # Using actual puzzles would cause extremely long compilation times due to
    # tracing all possible functions. Empty inputs allow JAX to specialize the
    # compiled code without processing complex puzzle structures.
    astar_fn(inital_search_result, empty_states, filled, empty_target)

    if show_compile_time:
        end = time.time()
        print(f"Compile Time: {end - start:6.2f} seconds")
        print("JIT compiled\n\n")

    return search_result_build, astar_fn
