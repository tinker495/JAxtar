import time

import chex
import jax
import jax.numpy as jnp
from Xtructure import HashTable, hash_func_builder

from heuristic.heuristic_base import Heuristic
from JAxtar.annotate import ACTION_DTYPE, KEY_DTYPE, SIZE_DTYPE
from JAxtar.search_base import Current, Current_with_Parent, Parent, SearchResult
from JAxtar.util import (
    flatten_array,
    flatten_tree,
    set_array_as_condition,
    unflatten_array,
    unflatten_tree,
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
    Builds and returns a JAX-accelerated A* search function.

    Args:
        puzzle: Puzzle instance that defines the problem space and operations.
        heuristic: Heuristic instance that provides state evaluation.
        batch_size: Number of states to process in parallel (default: 1024).
        max_nodes: Maximum number of nodes to explore before terminating (default: 1e6).
        cost_weight: Weight applied to the path cost in f(n) = g(n) + w*h(n) (default: 1.0-1e-6).
                    Values closer to 1.0 make the search more greedy/depth-first.
        show_compile_time: If True, displays the time taken to compile the search function (default: False).

    Returns:
        A function that performs A* search given a start state and solve configuration.
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
            _,
            _,
            idx,
            table_idx,
        ) = search_result.hashtable.parallel_insert(hash_func, states, filled)

        cost = jnp.where(filled, 0, jnp.inf)
        search_result.cost = set_array_as_condition(
            search_result.cost,
            filled,
            cost,
            idx,
            table_idx,
        )
        hash_idxs = Current(index=idx, table_index=table_idx, cost=cost)

        def _cond(input: tuple[SearchResult, Current, chex.Array]):
            search_result, parent, filled = input
            hash_size = search_result.generated_size
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
            parent_action = jnp.tile(
                jnp.expand_dims(jnp.arange(ncost.shape[0], dtype=ACTION_DTYPE), axis=1),
                (1, ncost.shape[1]),
            )  # [n_neighbours, batch_size]
            nextcosts = (cost[jnp.newaxis, :] + ncost).astype(
                KEY_DTYPE
            )  # [n_neighbours, batch_size]
            filleds = jnp.isfinite(nextcosts)  # [n_neighbours, batch_size]
            parent_index = jnp.tile(
                jnp.expand_dims(jnp.arange(ncost.shape[1]), axis=0), (ncost.shape[0],)
            )  # [n_neighbours, batch_size]

            flatten_neighbours = flatten_tree(neighbours, 2)
            flatten_filleds = flatten_array(filleds, 2)
            flatten_nextcosts = flatten_array(nextcosts, 2)
            flatten_parent_index = flatten_array(parent_index, 2)
            flatten_parent_action = flatten_array(parent_action, 2)
            (
                search_result.hashtable,
                flatten_inserted,
                _,
                idxs,
                table_idxs,
            ) = search_result.hashtable.parallel_insert(
                hash_func, flatten_neighbours, flatten_filleds
            )

            argsort_idx = jnp.argsort(flatten_inserted, axis=0)  # sort by inserted

            flatten_inserted = flatten_inserted[argsort_idx]
            flatten_neighbours = flatten_neighbours[argsort_idx]
            flatten_nextcosts = flatten_nextcosts[argsort_idx]
            flatten_parent_index = flatten_parent_index[argsort_idx]
            flatten_parent_action = flatten_parent_action[argsort_idx]

            idxs = idxs[argsort_idx]
            table_idxs = table_idxs[argsort_idx]

            idxs = unflatten_array(idxs, filleds.shape)
            table_idxs = unflatten_array(table_idxs, filleds.shape)
            nextcosts = unflatten_array(flatten_nextcosts, filleds.shape)
            current = Current(index=idxs, table_index=table_idxs, cost=nextcosts)
            parent_indexs = unflatten_array(flatten_parent_index, filleds.shape)
            parent_action = unflatten_array(flatten_parent_action, filleds.shape)
            neighbours = unflatten_tree(flatten_neighbours, filleds.shape)
            inserted = unflatten_array(flatten_inserted, filleds.shape)

            def _inserted(search_result: SearchResult, neighbour, current, inserted):
                neighbour_heur = heuristic.batched_distance(solve_config, neighbour).astype(
                    KEY_DTYPE
                )
                # cache the heuristic value
                search_result.dist = set_array_as_condition(
                    search_result.dist,
                    inserted,
                    neighbour_heur,
                    current.index,
                    current.table_index,
                )
                return search_result, neighbour_heur

            def _not_inserted(search_result: SearchResult, neighbour, current, inserted):
                # get cached heuristic value
                neighbour_heur = search_result.get_dist(current)
                return search_result, neighbour_heur

            def _scan(search_result: SearchResult, val):
                neighbour, parent_action, current, inserted, parent_index = val

                search_result, neighbour_heur = jax.lax.cond(
                    jnp.any(inserted),
                    _inserted,
                    _not_inserted,
                    search_result,
                    neighbour,
                    current,
                    inserted,
                )
                neighbour_key = (cost_weight * current.cost + neighbour_heur).astype(KEY_DTYPE)

                optimal = jnp.less(current.cost, search_result.get_cost(current))
                neighbour_key = jnp.where(optimal, neighbour_key, jnp.inf)

                aranged_parent = parent[parent_index]
                vals = Current_with_Parent(
                    current=current,
                    parent=Parent(
                        action=parent_action,
                        index=aranged_parent.index,
                        table_index=aranged_parent.table_index,
                    ),
                )

                search_result.priority_queue = search_result.priority_queue.insert(
                    neighbour_key,
                    vals,
                    added_size=jnp.sum(jnp.isfinite(neighbour_key), dtype=SIZE_DTYPE),
                )
                return search_result, None

            search_result, _ = jax.lax.scan(
                _scan,
                search_result,
                (neighbours, parent_action, current, inserted, parent_indexs),
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
