import time

import chex
import jax
import jax.numpy as jnp

from heuristic.heuristic_base import Heuristic
from JAxtar.annotate import ACTION_DTYPE, KEY_DTYPE, SIZE_DTYPE
from JAxtar.hash import HashTable, hash_func_builder
from JAxtar.search_base import Current, Current_with_Parent, Parent, SearchResult
from JAxtar.util import (
    flatten_array,
    flatten_tree,
    set_array_as_condition,
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

    hash_func = hash_func_builder(statecls)

    def astar(
        solve_config: Puzzle.SolveConfig,
        start: Puzzle.State,
    ) -> tuple[SearchResult, chex.Array]:
        """
        astar is the implementation of the A* algorithm.
        """
        search_result: SearchResult = SearchResult.build(statecls, batch_size, max_nodes)
        states, filled = HashTable.make_batched(puzzle.State, start[jnp.newaxis, ...], batch_size)

        (
            search_result.hashtable,
            inserted,
            _,
            idx,
            table_idx,
        ) = search_result.hashtable.parallel_insert(hash_func, states, filled)

        cost = jnp.where(filled, 0, jnp.inf)
        search_result.cost = set_array_as_condition(
            search_result.cost,
            inserted,
            cost,
            idx,
            table_idx,
        )
        hash_idxs = Current(index=idx, table_index=table_idx, cost=cost)

        def _cond(input: tuple[SearchResult, Current, chex.Array]):
            search_result, parent, filled = input
            hash_size = search_result.hashtable.size
            size_cond1 = filled.any()  # queue is not empty
            size_cond2 = hash_size < max_nodes  # hash table is not full
            size_cond = jnp.logical_and(size_cond1, size_cond2)

            states = search_result.get_state(parent)
            solved = puzzle.batched_is_solved(solve_config, states)
            return jnp.logical_and(size_cond, ~solved.any())

        def _body(input: tuple[SearchResult, Current, chex.Array]):
            search_result, parent, filled = input

            cost = search_result.get_cost(parent)
            states = search_result.get_state(parent)

            neighbours, ncost = puzzle.batched_get_neighbours(solve_config, states, filled)
            parent_action = jnp.arange(ncost.shape[0], dtype=ACTION_DTYPE)
            nextcosts = cost[jnp.newaxis, :] + ncost  # [n_neighbours, batch_size]
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
            current = Current(index=idxs, table_index=table_idxs, cost=nextcosts)

            def _scan(search_result: SearchResult, val):
                neighbour, parent_action, current = val
                neighbour_heur = heuristic.batched_distance(solve_config, neighbour)
                neighbour_key = (cost_weight * current.cost + neighbour_heur).astype(KEY_DTYPE)

                optimal = jnp.less(
                    current.cost, search_result.cost[current.index, current.table_index]
                )
                neighbour_key = jnp.where(optimal, neighbour_key, jnp.inf)

                parent_action = jnp.tile(parent_action, (neighbour_key.shape[0],))
                vals = Current_with_Parent(
                    current=current,
                    parent=Parent(
                        action=parent_action, index=parent.index, table_index=parent.table_index
                    ),
                )

                search_result.priority_queue = search_result.priority_queue.insert(
                    neighbour_key,
                    vals,
                    added_size=jnp.sum(optimal, dtype=SIZE_DTYPE),
                )
                return search_result, None

            search_result, _ = jax.lax.scan(
                _scan,
                search_result,
                (neighbours, parent_action, current),
            )
            search_result, parent, filled = search_result.pop_full()
            return search_result, parent, filled

        (search_result, idxes, filled) = jax.lax.while_loop(
            _cond, _body, (search_result, hash_idxs, filled)
        )
        states = search_result.get_state(idxes)
        solved = puzzle.batched_is_solved(solve_config, states)
        search_result.solved = solved.any()
        search_result.solved_idx = idxes[jnp.argmax(solved)]
        return search_result

    astar_fn = jax.jit(astar)
    empty_solve_config = puzzle.SolveConfig.default()
    empty_states = puzzle.State.default()

    if show_compile_time:
        print("initializing jit")
        start = time.time()

    # Pass empty states and target to JIT-compile the function with simple data.
    # Using actual puzzles would cause extremely long compilation times due to
    # tracing all possible functions. Empty inputs allow JAX to specialize the
    # compiled code without processing complex puzzle structures.
    astar_fn(empty_solve_config, empty_states)

    if show_compile_time:
        end = time.time()
        print(f"Compile Time: {end - start:6.2f} seconds")
        print("JIT compiled\n\n")

    return astar_fn
