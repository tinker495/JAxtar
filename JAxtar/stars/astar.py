from typing import Any

import jax
import jax.numpy as jnp
from puxle import Puzzle

from helpers.jax_compile import jit_with_warmup
from heuristic.heuristic_base import Heuristic
from JAxtar.annotate import MIN_BATCH_SIZE
from JAxtar.core.common import finalize_search_result
from JAxtar.core.expansion import EagerExpansion
from JAxtar.core.loop import unified_search_loop_builder
from JAxtar.core.result import Current, SearchResult
from JAxtar.core.scoring import AStarScoring
from JAxtar.utils.batch_switcher import variable_batch_switcher_builder


def astar_builder(
    puzzle: Puzzle,
    heuristic: Heuristic,
    batch_size: int = 1024,
    max_nodes: int = int(1e6),
    pop_ratio: float = jnp.inf,
    cost_weight: float = 1.0 - 1e-6,
    show_compile_time: bool = False,
    warmup_inputs: tuple[Puzzle.SolveConfig, Puzzle.State] | None = None,
):
    """
    Builds and returns a JAX-accelerated A* search function using Unified Core.

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
    # 1. Define Policies
    variable_heuristic_batch_switcher = variable_batch_switcher_builder(
        heuristic.batched_distance,
        max_batch_size=batch_size,
        min_batch_size=MIN_BATCH_SIZE,
        pad_value=jnp.inf,
    )
    scoring_policy = AStarScoring()
    expansion_policy = EagerExpansion(
        scoring_policy=scoring_policy,
        heuristic_fn=lambda p, s, m: variable_heuristic_batch_switcher(p, s, m),
        cost_weight=cost_weight,
    )

    # 2. Build Loop
    # Eager expansion uses Current type for PQ
    init_loop_state, loop_condition, loop_body = unified_search_loop_builder(
        puzzle,
        expansion_policy,
        batch_size,
        max_nodes,
        pop_ratio,
        min_pop=1,  # Default min_pop logic handled in SearchResult, but let's pass explicit if needed
        pq_val_type=Current,
    )

    def astar(
        solve_config: Puzzle.SolveConfig,
        start: Puzzle.State,
        **kwargs: Any,
    ) -> SearchResult:
        """
        astar is the implementation of the A* algorithm (Unified).
        """
        # Prepare heuristic
        heuristic_params = heuristic.prepare_heuristic_parameters(solve_config, **kwargs)

        loop_state = init_loop_state(solve_config, start, heuristic_params=heuristic_params)

        loop_state = jax.lax.while_loop(loop_condition, loop_body, loop_state)

        search_result = loop_state.search_result
        current = loop_state.current
        # states = loop_state.states # No states on LoopState currently?
        # Check unified_search_loop_builder in core/loop.py:
        # LoopState definition:
        # current: Any
        # filled: Array

        # We need states to check for solution if not already checked inside loop?
        # `loop_condition` checks solution. `loop_continue_if_not_solved` logic.
        # But `finalize_search_result` usually needs `solved_mask`.

        # We retrieve states from `current`
        states = search_result.get_state(current)  # This works for Eager (Current type)
        filled = loop_state.filled

        solved_mask = jnp.logical_and(puzzle.batched_is_solved(solve_config, states), filled)
        return finalize_search_result(search_result, current, solved_mask)

    return jit_with_warmup(
        astar,
        puzzle=puzzle,
        show_compile_time=show_compile_time,
        warmup_inputs=warmup_inputs,
    )
