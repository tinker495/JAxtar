import time

import chex
import jax
import jax.numpy as jnp
import xtructure.numpy as xnp
from puxle import Puzzle

from JAxtar.annotate import ACTION_DTYPE, KEY_DTYPE, MIN_BATCH_SIZE
from JAxtar.stars.search_base import HashIdx, Parant_with_Costs, Parent, SearchResult
from JAxtar.utils.batch_switcher import variable_batch_switcher_builder
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
        pop_ratio: Ratio of states to pop from the priority queue.
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
    variable_q_batch_switcher = variable_batch_switcher_builder(
        lambda solve_config, current: q_fn.batched_q_value(solve_config, current),
        max_batch_size=batch_size,
        min_batch_size=MIN_BATCH_SIZE,
        pad_value=jnp.inf,
    )
    denom = max(1, puzzle.action_size // 2)
    min_pop = max(1, MIN_BATCH_SIZE // denom)

    def qstar(
        solve_config: Puzzle.SolveConfig,
        start: Puzzle.State,
    ) -> SearchResult:
        """
        qstar is the implementation of the Q* algorithm.
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

            # Compute Q-values for parent states (not neighbors)
            # This gives us Q(s, a) for all actions from parent states
            q_vals = variable_q_batch_switcher(solve_config, states, filled)
            q_vals = q_vals.transpose().astype(KEY_DTYPE)  # [action_size, batch_size]

            neighbour_keys = (cost_weight * costs + q_vals).astype(KEY_DTYPE)
            neighbour_keys = jnp.where(filled_tiles, neighbour_keys, jnp.inf)

            vals = Parant_with_Costs(
                parent=Parent(hashidx=idx_tiles, action=action),
                cost=costs,
                dist=q_vals,
            )

            # Look-a-head pruning
            neighbour_look_a_head, ncosts = puzzle.batched_get_neighbours(
                solve_config, states, filled
            )  # [action_size, batch_size]
            look_a_head_costs = costs + ncosts  # [action_size, batch_size]
            flattened_neighbour_look_head = neighbour_look_a_head.flatten()
            flattened_look_a_head_costs = look_a_head_costs.flatten()
            flattened_filled_tiles = filled_tiles.flatten()
            flattened_vals = vals.flatten()
            flattened_keys = neighbour_keys.flatten()

            current_hash_idxs, _ = search_result.hashtable.lookup_parallel(
                flattened_neighbour_look_head
            )

            old_costs = search_result.get_cost(current_hash_idxs)
            optimal_mask = jnp.less(flattened_look_a_head_costs, old_costs) & flattened_filled_tiles
            optimal_unique_mask = (
                xnp.unique_mask(
                    flattened_neighbour_look_head, flattened_look_a_head_costs, optimal_mask
                )
                & optimal_mask
            )

            flattened_neighbour_keys = jnp.where(optimal_unique_mask, flattened_keys, jnp.inf)

            # Sort to keep best candidates
            sorted_key, sorted_idx = jax.lax.sort_key_val(
                flattened_neighbour_keys, jnp.arange(flattened_neighbour_keys.shape[-1])
            )
            sorted_vals = flattened_vals[sorted_idx]
            sorted_optimal_unique_mask = optimal_unique_mask[sorted_idx]

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
