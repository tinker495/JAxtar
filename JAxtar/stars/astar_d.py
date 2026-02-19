from typing import Any

import jax
import jax.numpy as jnp
from puxle import Puzzle

from helpers.jax_compile import jit_with_warmup
from heuristic.heuristic_base import Heuristic
from JAxtar.annotate import MIN_BATCH_SIZE
from JAxtar.core.common import finalize_search_result
from JAxtar.core.expansion import DeferredExpansion
from JAxtar.core.loop import unified_search_loop_builder
from JAxtar.core.result import ParentWithCosts, SearchResult
from JAxtar.core.scoring import AStarScoring
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
    warmup_inputs: tuple[Puzzle.SolveConfig, Puzzle.State] | None = None,
):
    """
    Builds and returns a JAX-accelerated A* with deferred node evaluation (A* deferred).
    Uses Unified Core Architecture.
    """
    # 0. Build Switcher for efficiency
    variable_heuristic_batch_switcher = variable_batch_switcher_builder(
        heuristic.batched_distance,
        max_batch_size=batch_size,
        min_batch_size=MIN_BATCH_SIZE,
        pad_value=jnp.inf,
    )

    # 1. Define Policies
    scoring_policy = AStarScoring()
    expansion_policy = DeferredExpansion(
        scoring_policy=scoring_policy,
        heuristic_fn=lambda p, s, m: variable_heuristic_batch_switcher(p, s, m),
        cost_weight=cost_weight,
        look_ahead_pruning=look_ahead_pruning,
    )

    # 2. Build Loop
    # Deferred uses ParentWithCosts for PQ
    denom = max(1, puzzle.action_size // 2)
    min_pop = max(1, MIN_BATCH_SIZE // denom)
    init_loop_state, loop_condition, loop_body = unified_search_loop_builder(
        puzzle,
        expansion_policy,
        batch_size,
        max_nodes,
        pop_ratio,
        min_pop=min_pop,
        pq_val_type=ParentWithCosts,
    )

    def astar_d(
        solve_config: Puzzle.SolveConfig,
        start: Puzzle.State,
        **kwargs: Any,
    ) -> SearchResult:
        """
        astar_d is the implementation of the A* with deferred search algorithm.
        """
        # Prepare heuristic
        heuristic_params = heuristic.prepare_heuristic_parameters(solve_config, **kwargs)

        loop_state = init_loop_state(solve_config, start, heuristic_params=heuristic_params)

        loop_state = jax.lax.while_loop(loop_condition, loop_body, loop_state)

        search_result = loop_state.search_result
        current = loop_state.current
        # Retrieve states from current (which is Current type here because pop_full returns Current)
        # Note: Deferred pop returns Current.
        states = search_result.get_state(current)
        filled = loop_state.filled

        solved_mask = jnp.logical_and(puzzle.batched_is_solved(solve_config, states), filled)
        return finalize_search_result(search_result, current, solved_mask)

    return jit_with_warmup(
        astar_d,
        puzzle=puzzle,
        show_compile_time=show_compile_time,
        warmup_inputs=warmup_inputs,
    )
