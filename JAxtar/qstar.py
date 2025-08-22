import time

import chex
import jax
import jax.numpy as jnp
import xtructure.numpy as xnp
from puxle import Puzzle

from JAxtar.annotate import ACTION_DTYPE, KEY_DTYPE
from JAxtar.search_base import Current, Current_with_Parent, Parent, SearchResult
from qfunction.q_base import QFunction


def qstar_builder(
    puzzle: Puzzle,
    q_fn: QFunction,
    batch_size: int = 1024,
    max_nodes: int = int(1e6),
    pop_ratio: float = jnp.inf,
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

    # min_pop determines the minimum number of states to pop from the priority queue in each batch.
    # This value is set to optimize the efficiency of batched operations.
    # By ensuring that at least this many states are processed together,
    # we maximize parallelism and hardware utilization,
    # which is especially important for JAX and accelerator-based computation.
    # The formula (batch_size // (puzzle.action_size // 2)) is chosen to balance the number of expansions per batch,
    # so that each batch is filled as evenly as possible and computational resources are used efficiently.
    denom = max(1, puzzle.action_size // 2)
    min_pop = max(1, batch_size // denom)

    def qstar(
        solve_config: Puzzle.SolveConfig,
        start: Puzzle.State,
    ) -> SearchResult:
        """
        qstar is the implementation of the Q* algorithm.
        """
        search_result: SearchResult = SearchResult.build(
            statecls, batch_size, max_nodes, pop_ratio=pop_ratio, min_pop=min_pop
        )

        (
            search_result.hashtable,
            _,
            hash_idx,
        ) = search_result.hashtable.insert(start)

        search_result.cost = search_result.cost.at[hash_idx.index].set(0)
        hash_idxs = Current(hashidx=hash_idx, cost=jnp.zeros((), dtype=KEY_DTYPE),)[
            jnp.newaxis
        ].padding_as_batch((batch_size,))
        filled = jnp.zeros(batch_size, dtype=jnp.bool_).at[0].set(True)

        def _cond(input: tuple[SearchResult, Current, chex.Array]):
            search_result, parent, filled = input
            hash_size = search_result.generated_size
            size_cond1 = filled.any()  # queue is not empty
            size_cond2 = hash_size < max_nodes  # hash table is not full
            size_cond = jnp.logical_and(size_cond1, size_cond2)

            states = search_result.get_state(parent)
            solved = puzzle.batched_is_solved(solve_config, states)
            solved = jnp.logical_and(solved, filled)
            return jnp.logical_and(size_cond, ~solved.any())

        def _body(input: tuple[SearchResult, Current, chex.Array]):
            search_result, parent, filled = input

            cost = search_result.get_cost(parent)
            states = search_result.get_state(parent)

            neighbours, ncost = puzzle.batched_get_neighbours(solve_config, states, filled)
            parent_action = jnp.tile(
                jnp.arange(ncost.shape[0], dtype=ACTION_DTYPE)[:, jnp.newaxis],
                (1, ncost.shape[1]),
            )  # [n_neighbours, batch_size]
            nextcosts = (cost[jnp.newaxis, :] + ncost).astype(
                KEY_DTYPE
            )  # [n_neighbours, batch_size]
            filleds = jnp.isfinite(nextcosts)  # [n_neighbours, batch_size]
            parent_index = jnp.tile(
                jnp.arange(ncost.shape[1], dtype=ACTION_DTYPE)[jnp.newaxis, :],
                (ncost.shape[0], 1),
            )  # [n_neighbours, batch_size]
            unflatten_shape = filleds.shape

            # Compute Q-values for parent states (not neighbors)
            # This gives us Q(s, a) for all actions from parent states
            q_vals = (
                q_fn.batched_q_value(solve_config, states).transpose().astype(KEY_DTYPE)
            )  # [batch_size, n_neighbours] -> [n_neighbours, batch_size]

            flatten_neighbours = neighbours.flatten()
            flatten_filleds = filleds.flatten()
            flatten_nextcosts = nextcosts.flatten()
            flatten_parent_index = parent_index.flatten()
            flatten_parent_action = parent_action.flatten()
            flatten_q_vals = q_vals.flatten()
            (
                search_result.hashtable,
                _,
                cheapest_uniques_mask,
                hash_idx,
            ) = search_result.hashtable.parallel_insert(
                flatten_neighbours, flatten_filleds, flatten_nextcosts
            )

            # It must also be cheaper than any previously found path to this state.
            optimal_mask = jnp.less(flatten_nextcosts, search_result.get_cost(hash_idx))

            # Combine all conditions for the final decision.
            final_process_mask = jnp.logical_and(cheapest_uniques_mask, optimal_mask)

            # Update the cost (g-value) for the newly found optimal paths before they are
            # masked out. This ensures the cost table is always up-to-date.
            search_result.cost = xnp.update_on_condition(
                search_result.cost,
                hash_idx.index,
                final_process_mask,
                flatten_nextcosts,  # Use costs before they are set to inf
            )

            # Apply the final mask: deactivate non-optimal nodes by setting their cost to infinity
            # and updating the insertion flag. This ensures they are ignored in subsequent steps.
            flatten_nextcosts = jnp.where(final_process_mask, flatten_nextcosts, jnp.inf)
            # Stable partition to group useful entries first.
            # Improves computational efficiency by gathering only batches with samples that need updates.
            invperm = jnp.argsort(final_process_mask)

            flatten_final_process_mask = final_process_mask[invperm]
            flatten_neighbours = flatten_neighbours[invperm]
            flatten_nextcosts = flatten_nextcosts[invperm]
            flatten_q_vals = flatten_q_vals[invperm]
            flatten_parent_index = flatten_parent_index[invperm]
            flatten_parent_action = flatten_parent_action[invperm]

            hash_idx = hash_idx[invperm]

            hash_idx = hash_idx.reshape(unflatten_shape)
            nextcosts = flatten_nextcosts.reshape(unflatten_shape)
            q_vals = flatten_q_vals.reshape(unflatten_shape)
            current = Current(hashidx=hash_idx, cost=nextcosts)
            parent_indexs = flatten_parent_index.reshape(unflatten_shape)
            parent_action = flatten_parent_action.reshape(unflatten_shape)
            neighbours = flatten_neighbours.reshape(unflatten_shape)
            final_process_mask = flatten_final_process_mask.reshape(unflatten_shape)

            def _insert(search_result: SearchResult, current, q_vals, parent_index, parent_action):
                neighbour_key = (cost_weight * current.cost + q_vals).astype(KEY_DTYPE)

                aranged_parent = parent[parent_index]
                vals = Current_with_Parent(
                    current=current,
                    parent=Parent(
                        action=parent_action,
                        hashidx=aranged_parent.hashidx,
                    ),
                )

                search_result.priority_queue = search_result.priority_queue.insert(
                    neighbour_key,
                    vals,
                )
                return search_result

            def _scan(search_result: SearchResult, val):
                parent_action, current, q_vals, final_process_mask, parent_index = val

                search_result = jax.lax.cond(
                    jnp.any(final_process_mask),
                    _insert,
                    lambda search_result, *args: search_result,
                    search_result,
                    current,
                    q_vals,
                    parent_index,
                    parent_action,
                )
                return search_result, None

            search_result, _ = jax.lax.scan(
                _scan,
                search_result,
                (parent_action, current, q_vals, final_process_mask, parent_indexs),
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
