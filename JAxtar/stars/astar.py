import time

import chex
import jax
import jax.numpy as jnp
import xtructure.numpy as xnp
from puxle import Puzzle

from heuristic.heuristic_base import Heuristic
from JAxtar.annotate import ACTION_DTYPE, KEY_DTYPE, MIN_BATCH_SIZE
from JAxtar.stars.search_base import Current, Current_with_Parent, Parent, SearchResult
from JAxtar.utils.batch_switcher import variable_batch_switcher_builder


def astar_builder(
    puzzle: Puzzle,
    heuristic: Heuristic,
    batch_size: int = 1024,
    max_nodes: int = int(1e6),
    pop_ratio: float = jnp.inf,
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

    # min_pop determines the minimum number of states to pop from the priority queue in each batch.
    # This value is set to optimize the efficiency of batched operations.
    # By ensuring that at least this many states are processed together,
    # we maximize parallelism and hardware utilization,
    # which is especially important for JAX and accelerator-based computation.
    # The formula (batch_size // (puzzle.action_size // 2)) is chosen to balance the number of expansions per batch,
    # so that each batch is filled as evenly as possible and computational resources are used efficiently.
    variable_heuristic_batch_switcher = variable_batch_switcher_builder(
        lambda solve_config, current: heuristic.batched_distance(solve_config, current),
        max_batch_size=batch_size,
        min_batch_size=MIN_BATCH_SIZE,
        pad_value=jnp.inf,
    )
    denom = max(1, puzzle.action_size // 2)
    min_pop = max(1, MIN_BATCH_SIZE // denom)

    def astar(
        solve_config: Puzzle.SolveConfig,
        start: Puzzle.State,
    ) -> SearchResult:
        """
        astar is the implementation of the A* algorithm.
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

            flatten_neighbours = neighbours.flatten()
            flatten_filleds = filleds.flatten()
            flatten_nextcosts = nextcosts.flatten()
            flatten_parent_index = parent_index.flatten()
            flatten_parent_action = parent_action.flatten()
            (
                search_result.hashtable,
                flatten_new_states_mask,
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
            invperm = stable_partition_three(flatten_new_states_mask, final_process_mask)

            flatten_final_process_mask = final_process_mask[invperm]
            flatten_new_states_mask = flatten_new_states_mask[invperm]
            flatten_neighbours = flatten_neighbours[invperm]
            flatten_nextcosts = flatten_nextcosts[invperm]
            flatten_parent_index = flatten_parent_index[invperm]
            flatten_parent_action = flatten_parent_action[invperm]

            hash_idx = hash_idx[invperm]
            current = Current(hashidx=hash_idx, cost=flatten_nextcosts)

            flatten_aranged_parent = parent[flatten_parent_index]
            flatten_vals = Current_with_Parent(
                current=current,
                parent=Parent(
                    action=flatten_parent_action,
                    hashidx=flatten_aranged_parent.hashidx,
                ),
            )

            vals = flatten_vals.reshape(unflatten_shape)
            neighbours = flatten_neighbours.reshape(unflatten_shape)
            new_states_mask = flatten_new_states_mask.reshape(unflatten_shape)
            final_process_mask = flatten_final_process_mask.reshape(unflatten_shape)

            def _new_states(search_result: SearchResult, vals, neighbour, new_states_mask):
                neighbour_heur = variable_heuristic_batch_switcher(
                    solve_config, neighbour, new_states_mask
                ).astype(KEY_DTYPE)
                # cache the heuristic value
                search_result.dist = xnp.update_on_condition(
                    search_result.dist,
                    vals.current.hashidx.index,
                    new_states_mask,
                    neighbour_heur,
                )
                return search_result, neighbour_heur

            def _old_states(search_result: SearchResult, vals, neighbour, new_states_mask):
                neighbour_heur = search_result.dist[vals.current.hashidx.index]
                return search_result, neighbour_heur

            def _inserted(
                search_result: SearchResult,
                vals,
                neighbour_heur,
            ):
                neighbour_key = (cost_weight * vals.current.cost + neighbour_heur).astype(KEY_DTYPE)

                search_result.priority_queue = search_result.priority_queue.insert(
                    neighbour_key,
                    vals,
                )
                return search_result

            def _scan(search_result: SearchResult, val):
                vals, neighbour, new_states_mask, final_process_mask = val

                search_result, neighbour_heur = jax.lax.cond(
                    jnp.any(new_states_mask),
                    _new_states,
                    _old_states,
                    search_result,
                    vals,
                    neighbour,
                    new_states_mask,
                )

                search_result = jax.lax.cond(
                    jnp.any(final_process_mask),
                    _inserted,
                    lambda search_result, *args: search_result,
                    search_result,
                    vals,
                    neighbour_heur,
                )
                return search_result, None

            search_result, _ = jax.lax.scan(
                _scan,
                search_result,
                (vals, neighbours, new_states_mask, final_process_mask),
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


def stable_partition_three(mask2: chex.Array, mask1: chex.Array) -> chex.Array:
    """
    Compute a stable 3-way partition inverse permutation for flattened arrays.

    - Category 2 (mask2): first block
    - Category 1 (mask1 & ~mask2): second block
    - Category 0 (else): last block

    Returns indices suitable for gathering flattened arrays to achieve the
    [2..., 1..., 0...] ordering while preserving relative order within each class.
    """

    # Flatten masks
    flat2 = mask2.reshape(-1)
    # Ensure category 1 excludes category 2
    flat1 = jnp.logical_and(mask1.reshape(-1), jnp.logical_not(flat2))

    # Compute category id per element: 2, 1, or 0
    cat = jnp.where(flat2, 2, jnp.where(flat1, 1, 0)).astype(jnp.int32)

    n = cat.shape[0]
    indices = jnp.arange(n, dtype=jnp.int32)

    # Stable sort by key = -cat so that 2-block comes first, then 1, then 0.
    # The stable flag preserves original order within equal keys (intra-class stability).
    _, invperm = jax.lax.sort_key_val(-cat, indices, dimension=0, is_stable=True)

    # Return gather indices: arr[invperm] yields [2..., 1..., 0...] with stable intra-class order
    return invperm
