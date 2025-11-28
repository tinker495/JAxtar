import time

import chex
import jax
import jax.numpy as jnp
import xtructure.numpy as xnp
from puxle import Puzzle

from heuristic.heuristic_base import Heuristic
from JAxtar.annotate import ACTION_DTYPE, KEY_DTYPE, MIN_BATCH_SIZE
from JAxtar.stars.search_base import HashIdx, Parant_with_Costs, Parent, SearchResult
from JAxtar.utils.batch_switcher import variable_batch_switcher_builder


def astar_d_builder(
    puzzle: Puzzle,
    heuristic: Heuristic,
    batch_size: int = 1024,
    max_nodes: int = int(1e6),
    pop_ratio: float = jnp.inf,
    cost_weight: float = 1.0 - 1e-6,
    show_compile_time: bool = False,
    look_ahead_pruning: bool = True,
):
    """
    Builds and returns a JAX-accelerated A* with deferred node evaluation (A* deferred).

    In standard A*, when a node is expanded, all its children are generated and their heuristics
    are evaluated immediately to calculate f(n) = g(n) + h(n) for insertion into the open list.

    In A* deferred, we delay the generation and heuristic evaluation of children. Instead,
    when a node is expanded, we insert its potential *actions* (edges) into the priority queue,
    using the *parent's* f-value (or similar) as the priority.
    Only when an action is popped from the queue do we actually generate the child state
    and evaluate its heuristic (if it becomes a parent for the next step).

    This approach is beneficial when:
    1. The heuristic calculation is expensive.
    2. The branching factor is large (avoids evaluating children that are never explored).
    3. We want to maximize batch efficiency in JAX by keeping the pipeline uniform.

    Args:
        puzzle: Puzzle instance that defines the problem space and operations.
        heuristic: Heuristic instance that provides state evaluation.
        batch_size: Number of states to process in parallel (default: 1024).
        max_nodes: Maximum number of nodes to explore before terminating (default: 1e6).
        pop_ratio: Ratio of states to pop from the priority queue.
        cost_weight: Weight applied to the path cost in the A* with deferred search algorithm (default: 1.0-1e-6).
                    Values closer to 1.0 make the search more greedy/depth-first.
        show_compile_time: If True, displays the time taken to compile the search function (default: False).
        look_ahead_pruning: Enables neighbour look-ahead pruning (default: True). Disable
            to recover canonical A* deferred behaviour when simulator cost outweighs queue
            pressure.

    Returns:
        A function that performs A* with deferred search given a start state and solve configuration.
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

    def astar_d(
        solve_config: Puzzle.SolveConfig,
        start: Puzzle.State,
    ) -> SearchResult:
        """
        astar_d is the implementation of the A* with deferred search algorithm.
        """
        search_result: SearchResult = SearchResult.build(
            statecls,
            batch_size,
            max_nodes,
            pop_ratio=pop_ratio,
            min_pop=min_pop,
            parant_with_costs=True,
        )

        (
            search_result.hashtable,
            _,
            hash_idx,
        ) = search_result.hashtable.insert(start)

        search_result.cost = search_result.cost.at[hash_idx.index].set(0)
        costs = jnp.zeros((batch_size,), dtype=KEY_DTYPE)
        states = xnp.pad(start, (0, batch_size - 1))
        hash_idxs = xnp.pad(hash_idx, (0, batch_size - 1))
        filled = jnp.zeros(batch_size, dtype=jnp.bool_).at[0].set(True)

        def _cond(input: tuple[SearchResult, jnp.ndarray, Puzzle.State, HashIdx, chex.Array]):
            search_result, _, states, _, filled = input
            hash_size = search_result.generated_size
            size_cond1 = filled.any()  # queue is not empty
            size_cond2 = hash_size < max_nodes  # hash table is not full
            size_cond = jnp.logical_and(size_cond1, size_cond2)

            solved = puzzle.batched_is_solved(solve_config, states)
            solved = jnp.logical_and(solved, filled)
            return jnp.logical_and(size_cond, ~solved.any())

        def _body(input: tuple[SearchResult, jnp.ndarray, Puzzle.State, HashIdx, chex.Array]):
            search_result, cost, states, hash_idx, filled = input

            idx_tiles = xnp.tile(
                hash_idx, (puzzle.action_size, 1)
            )  # [action_size, batch_size, ...]
            action = jnp.tile(
                jnp.arange(puzzle.action_size, dtype=ACTION_DTYPE)[:, jnp.newaxis],
                (1, cost.shape[0]),
            )  # [n_neighbours, batch_size]
            costs = jnp.tile(
                cost[jnp.newaxis, :], (puzzle.action_size, 1)
            )  # [action_size, batch_size]
            filled_tiles = jnp.tile(
                filled[jnp.newaxis, :], (puzzle.action_size, 1)
            )  # [action_size, batch_size]

            flattened_filled_tiles = filled_tiles.flatten()

            if look_ahead_pruning:
                # NOTE: Standard A* deferred evaluates children only when popped.
                # This optional look-ahead peeks at concrete neighbours to cull
                # duplicates and dominated costs ahead of time, which means it is
                # no longer a textbook implementation. It pays off when state
                # transitions are cheap relative to priority-queue traffic and in
                # highly reversible puzzles that re-visit the same states often.
                neighbour_look_a_head, ncosts = puzzle.batched_get_neighbours(
                    solve_config, states, filled
                )  # [action_size, batch_size]
                look_a_head_costs = costs + ncosts  # [action_size, batch_size]

                # Calculate heuristics for look-ahead neighbours immediately
                # Use scan to process each action's batch separately to avoid shape mismatch
                def _calc_heuristic(carry, input_slice):
                    states_slice, filled_slice = input_slice
                    h_val = variable_heuristic_batch_switcher(
                        solve_config, states_slice, filled_slice
                    )
                    return carry, h_val

                _, heuristic_vals = jax.lax.scan(
                    _calc_heuristic, None, (neighbour_look_a_head, filled_tiles)
                )  # [action_size, batch_size]

                heuristic_vals = jnp.where(filled_tiles, heuristic_vals, jnp.inf).astype(KEY_DTYPE)

                flattened_neighbour_look_head = neighbour_look_a_head.flatten()
                flattened_look_a_head_costs = look_a_head_costs.flatten()

                current_hash_idxs, found = search_result.hashtable.lookup_parallel(
                    flattened_neighbour_look_head, flattened_filled_tiles
                )

                old_costs = search_result.get_cost(current_hash_idxs)

                # Only consider nodes that are either:
                # 1. Not found in the hash table (new nodes), or
                # 2. Found but have better cost than existing
                candidate_mask = jnp.logical_or(
                    ~found, jnp.less(flattened_look_a_head_costs, old_costs)
                )
                candidate_mask = candidate_mask & flattened_filled_tiles

                # Deduplicate within the batch
                optimal_mask = (
                    xnp.unique_mask(
                        flattened_neighbour_look_head, flattened_look_a_head_costs, candidate_mask
                    )
                    & candidate_mask
                )

                # f(n) = g(n) + h(n) using actual child costs and heuristics
                neighbour_keys = (cost_weight * look_a_head_costs + heuristic_vals).astype(
                    KEY_DTYPE
                )

            else:

                # Compute heuristic values for the states currently being expanded (parents).
                # Unlike standard A*, we calculate h(parent) here and use it to prioritize
                # the *actions* leading to its children. The children themselves are not
                # generated or evaluated yet.
                heuristic_vals = variable_heuristic_batch_switcher(
                    solve_config, states, filled
                )  # [batch_size]
                heuristic_vals = jnp.where(filled, heuristic_vals, jnp.inf)  # [batch_size]
                heuristic_vals = jnp.tile(
                    heuristic_vals[jnp.newaxis, :], (puzzle.action_size, 1)
                ).astype(
                    KEY_DTYPE
                )  # [action_size, batch_size]

                optimal_mask = flattened_filled_tiles

                # The priority for the actions is based on the parent's cost and heuristic.
                # f_action = g(parent) + h(parent) (approximately, modified by cost_weight)
                neighbour_keys = (cost_weight * costs + heuristic_vals).astype(KEY_DTYPE)

            vals = Parant_with_Costs(
                parent=Parent(hashidx=idx_tiles.flatten(), action=action.flatten()),
                cost=costs.flatten(),
                dist=heuristic_vals.flatten(),
            )
            flattened_vals = vals.flatten()
            flattened_keys = neighbour_keys.flatten()

            flattened_neighbour_keys = jnp.where(optimal_mask, flattened_keys, jnp.inf)

            # Sort to keep best candidates
            sorted_key, sorted_idx = jax.lax.sort_key_val(
                flattened_neighbour_keys, jnp.arange(flattened_neighbour_keys.shape[-1])
            )
            sorted_vals = flattened_vals[sorted_idx]
            sorted_optimal_unique_mask = optimal_mask[sorted_idx]

            neighbour_keys = sorted_key.reshape(puzzle.action_size, batch_size)
            vals = sorted_vals.reshape((puzzle.action_size, batch_size))
            optimal_unique_mask = sorted_optimal_unique_mask.reshape(puzzle.action_size, batch_size)

            def _insert(search_result: SearchResult, neighbour_keys, vals):

                search_result.priority_queue = search_result.priority_queue.insert(
                    neighbour_keys,
                    vals,
                )
                return search_result

            def _scan(search_result: SearchResult, val):
                neighbour_keys, vals, mask = val

                search_result = jax.lax.cond(
                    jnp.any(mask),
                    _insert,
                    lambda search_result, *args: search_result,
                    search_result,
                    neighbour_keys,
                    vals,
                )
                return search_result, None

            search_result, _ = jax.lax.scan(
                _scan,
                search_result,
                (neighbour_keys, vals, optimal_unique_mask),
            )
            search_result, min_val, next_states, filled = search_result.pop_full_with_actions(
                puzzle, solve_config
            )

            return search_result, min_val.cost, next_states, min_val.hashidx, filled

        (search_result, costs, states, hash_idxs, filled) = jax.lax.while_loop(
            _cond, _body, (search_result, costs, states, hash_idxs, filled)
        )
        solved = puzzle.batched_is_solved(solve_config, states)
        search_result.solved = solved.any()
        search_result.solved_idx = hash_idxs[jnp.argmax(solved)]
        return search_result

    astar_d_fn = jax.jit(astar_d)
    empty_solve_config = puzzle.SolveConfig.default()
    empty_states = puzzle.State.default()

    if show_compile_time:
        print("initializing jit")
        start = time.time()

    # Pass empty states and target to JIT-compile the function with simple data.
    # Using actual puzzles would cause extremely long compilation times due to
    # tracing all possible functions. Empty inputs allow JAX to specialize the
    # compiled code without processing complex puzzle structures.
    astar_d_fn(empty_solve_config, empty_states)

    if show_compile_time:
        end = time.time()
        print(f"Compile Time: {end - start:6.2f} seconds")
        print("JIT compiled\n\n")

    return astar_d_fn
