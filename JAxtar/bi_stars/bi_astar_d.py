from typing import Any

import jax
import jax.numpy as jnp
from puxle import Puzzle

from helpers.jax_compile import jit_with_warmup
from heuristic.heuristic_base import Heuristic
from JAxtar.annotate import MIN_BATCH_SIZE
from JAxtar.core.bi_loop import unified_bi_search_loop_builder
from JAxtar.core.bi_result import (
    BiDirectionalSearchResult,
    finalize_bidirectional_result,
)
from JAxtar.core.expansion import DeferredExpansion
from JAxtar.core.result import ParentWithCosts
from JAxtar.core.scoring import AStarScoring
from JAxtar.utils.batch_switcher import variable_batch_switcher_builder


def bi_astar_d_builder(
    puzzle: Puzzle,
    heuristic: Heuristic,
    batch_size: int = 1024,
    max_nodes: int = int(1e6),
    pop_ratio: float = jnp.inf,
    cost_weight: float = 1.0 - 1e-6,
    show_compile_time: bool = False,
    look_ahead_pruning: bool = True,
    terminate_on_first_solution: bool = True,
    inverse_action_map: jax.Array | None = None,
    warmup_inputs: tuple[Puzzle.SolveConfig, Puzzle.State] | None = None,
):
    """
    Builds and returns a JAX-accelerated Bidirectional A* with deferred node evaluation.
    Uses Unified Core Architecture.
    """
    # 0. Build Switcher
    variable_heuristic_batch_switcher = variable_batch_switcher_builder(
        heuristic.batched_distance,
        max_batch_size=batch_size,
        min_batch_size=MIN_BATCH_SIZE,
        pad_value=jnp.inf,
    )

    # 1. Define Policies
    scoring_policy = AStarScoring()

    # Forward Policy
    fwd_expansion = DeferredExpansion(
        scoring_policy=scoring_policy,
        heuristic_fn=lambda p, s, m: variable_heuristic_batch_switcher(p, s, m),
        cost_weight=cost_weight,
        look_ahead_pruning=look_ahead_pruning,
        is_backward=False,
    )

    # Backward Policy
    bwd_expansion = DeferredExpansion(
        scoring_policy=scoring_policy,
        heuristic_fn=lambda p, s, m: variable_heuristic_batch_switcher(p, s, m),
        cost_weight=cost_weight,
        look_ahead_pruning=look_ahead_pruning,
        is_backward=True,
        inverse_action_map=inverse_action_map,
    )

    # 2. Build Loop
    # Deferred uses ParentWithCosts for PQ
    init_loop_state, loop_condition, loop_body = unified_bi_search_loop_builder(
        puzzle,
        fwd_expansion,
        bwd_expansion,
        batch_size,
        max_nodes,
        pop_ratio,
        min_pop=1,
        pq_val_type=ParentWithCosts,
        terminate_on_first_solution=terminate_on_first_solution,
    )

    def bi_astar_d(
        solve_config: Puzzle.SolveConfig,
        start: Puzzle.State,
        **kwargs: Any,
    ) -> BiDirectionalSearchResult:
        """
        bi_astar_d implementation (Unified).
        """
        # Prepare heuristics
        heuristic_params = heuristic.prepare_heuristic_parameters(solve_config, **kwargs)

        # Inverse Config / Params
        hindsight_transform = getattr(puzzle, "hindsight_transform", None)
        if callable(hindsight_transform):
            inverse_solve_config = hindsight_transform(solve_config, start)
        else:
            get_inverse_solve_config = getattr(puzzle, "get_inverse_solve_config", None)
            if callable(get_inverse_solve_config):
                inverse_solve_config = get_inverse_solve_config(solve_config, start)
            else:
                # Fallback
                inverse_solve_config = solve_config.replace(TargetState=start)
        inverse_heuristic_params = heuristic.prepare_heuristic_parameters(
            inverse_solve_config, **kwargs
        )

        if hasattr(puzzle, "get_goal_state"):
            goal = puzzle.get_goal_state(solve_config)
        elif hasattr(solve_config, "TargetState"):
            goal = solve_config.TargetState
        else:
            raise AttributeError(
                "Puzzle does not support get_goal_state and solve_config has no TargetState."
            )

        loop_state = init_loop_state(
            solve_config,
            inverse_solve_config,
            start,
            goal,
            heuristic_params=heuristic_params,
            inverse_heuristic_params=inverse_heuristic_params,
        )

        loop_state = jax.lax.while_loop(loop_condition, loop_body, loop_state)

        return finalize_bidirectional_result(loop_state.bi_result)

    return jit_with_warmup(
        bi_astar_d,
        puzzle=puzzle,
        show_compile_time=show_compile_time,
        warmup_inputs=warmup_inputs,
    )
