import time

import chex
import jax
import jax.numpy as jnp

from JAxtar.annotate import ACTION_DTYPE, KEY_DTYPE
from JAxtar.search_base import (
    Current,
    Current_with_Parent,
    HashIdx,
    Parent,
    SearchResult,
)
from JAxtar.util import (
    flatten_array,
    flatten_tree,
    set_array_as_condition,
    unflatten_array,
)
from puzzle.puzzle_base import Puzzle
from qfunction.q_base import QFunction


def qstar_builder(
    puzzle: Puzzle,
    q_fn: QFunction,
    batch_size: int = 1024,
    max_nodes: int = int(1e6),
    cost_weight: float = 1.0 - 1e-6,
    show_compile_time: bool = False,
):
    """
    Builds and returns a JAX-accelerated Q* search function.

    Args:
        puzzle: Puzzle instance that defines the problem space and operations.
        q_fn: QFunction instance that provides state-action value estimation.
        batch_size: Number of states to process in parallel (default: 1024).
        max_nodes: Maximum number of nodes to explore before terminating (default: 1e6).
        cost_weight: Weight applied to the path cost in the Q* algorithm (default: 1.0-1e-6).
                    Values closer to 1.0 make the search more greedy/depth-first.
        show_compile_time: If True, displays the time taken to compile the search function (default: False).

    Returns:
        A function that performs Q* search given a start state and solve configuration.
    """

    statecls = puzzle.State

    def qstar(
        solve_config: Puzzle.SolveConfig,
        start: Puzzle.State,
    ) -> tuple[SearchResult, chex.Array]:
        """
        qstar is the implementation of the Q* algorithm.
        """
        search_result: SearchResult = SearchResult.build(statecls, batch_size, max_nodes)

        (
            search_result.hashtable,
            _,
            idx,
            table_idx,
        ) = search_result.hashtable.insert(start)

        search_result.cost = search_result.cost.at[idx, table_idx].set(0)
        hash_idxs = Current(
            hashidx=HashIdx(index=idx, table_index=table_idx),
            cost=jnp.zeros((), dtype=KEY_DTYPE),
        )[jnp.newaxis].padding_as_batch((batch_size,))
        filled = jnp.zeros(batch_size, dtype=jnp.bool_).at[0].set(True)

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
            parent_action = jnp.arange(ncost.shape[0], dtype=ACTION_DTYPE)
            nextcosts = (cost[jnp.newaxis, :] + ncost).astype(
                KEY_DTYPE
            )  # [n_neighbours, batch_size]
            filleds = jnp.isfinite(nextcosts)  # [n_neighbours, batch_size]
            q_vals = (
                q_fn.batched_q_value(solve_config, states).transpose().astype(KEY_DTYPE)
            )  # [batch_size, n_neighbours] -> [n_neighbours, batch_size]
            neighbour_key = (cost_weight * nextcosts + q_vals).astype(KEY_DTYPE)

            flatten_filleds = flatten_array(filleds, 2)
            flatten_q_vals = flatten_array(q_vals, 2)
            (
                search_result.hashtable,
                _,
                _,
                idxs,
                table_idxs,
            ) = search_result.hashtable.parallel_insert(
                flatten_tree(neighbours, 2), flatten_filleds
            )

            # cache the q value but this is not using in search
            search_result.dist = set_array_as_condition(
                search_result.dist,
                flatten_filleds,
                flatten_q_vals,
                idxs,
                table_idxs,
            )

            idxs = unflatten_array(idxs, filleds.shape)
            table_idxs = unflatten_array(table_idxs, filleds.shape)
            current = Current(hashidx=HashIdx(index=idxs, table_index=table_idxs), cost=nextcosts)

            def _scan(search_result: SearchResult, val):
                neighbour_key, parent_action, current = val

                optimal = jnp.less(current.cost, search_result.get_cost(current))
                neighbour_key = jnp.where(optimal, neighbour_key, jnp.inf)

                parent_action = jnp.tile(parent_action, (neighbour_key.shape[0],))
                vals = Current_with_Parent(
                    current=current,
                    parent=Parent(
                        action=parent_action,
                        hashidx=parent.hashidx,
                    ),
                )

                search_result.priority_queue = search_result.priority_queue.insert(
                    neighbour_key,
                    vals,
                )
                return search_result, None

            search_result, _ = jax.lax.scan(
                _scan,
                search_result,
                (neighbour_key, parent_action, current),
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

    qstar_fn = jax.jit(qstar)
    empty_solve_config = puzzle.SolveConfig.default()
    empty_states = puzzle.State.default()

    if show_compile_time:
        print("initializing jit")
        start = time.time()

    # Pass empty states and target to JIT-compile the function with simple data.
    # Using actual puzzles would cause extremely long compilation times due to
    # tracing all possible functions. Empty inputs allow JAX to specialize the
    # compiled code without processing complex puzzle structures.
    qstar_fn(empty_solve_config, empty_states)

    if show_compile_time:
        end = time.time()
        print(f"Compile Time: {end - start:6.2f} seconds")
        print("JIT compiled\n\n")

    return qstar_fn
