from typing import Any

import jax
import jax.numpy as jnp
from puxle import Puzzle

from helpers.jax_compile import jit_with_warmup
from heuristic.heuristic_base import Heuristic
from JAxtar.bi_stars.bi_search_base import (
    build_bi_search_result as _legacy_build_bi_search_result,
)
from JAxtar.core.bi_loop import unified_bi_search_loop_builder
from JAxtar.core.bi_result import (
    BiDirectionalSearchResult,
    finalize_bidirectional_result,
)
from JAxtar.core.expansion import EagerExpansion
from JAxtar.core.result import Current
from JAxtar.core.scoring import AStarScoring


def build_bi_search_result(*args, **kwargs):
    """Compatibility shim for tests/tools that monkeypatch this module symbol."""
    return _legacy_build_bi_search_result(*args, **kwargs)


def bi_astar_builder(
    puzzle: Puzzle,
    heuristic: Heuristic,
    batch_size: int = 1024,
    max_nodes: int = int(1e6),
    pop_ratio: float = jnp.inf,
    cost_weight: float = 1.0 - 1e-6,
    show_compile_time: bool = False,
    terminate_on_first_solution: bool = True,
    inverse_action_map: jax.Array | None = None,
    warmup_inputs: tuple[Puzzle.SolveConfig, Puzzle.State] | None = None,
):
    """
    Builds and returns a JAX-accelerated Bidirectional A* search function using Unified Core.
    """
    # 1. Define Policies
    scoring_policy = AStarScoring()

    # Forward Policy
    fwd_expansion = EagerExpansion(
        scoring_policy=scoring_policy,
        heuristic_fn=lambda p, s, m: heuristic.batched_distance(p, s),
        cost_weight=cost_weight,
        is_backward=False,
    )

    # Backward Policy (uses inverse_action_map optimization if provided)
    bwd_expansion = EagerExpansion(
        scoring_policy=scoring_policy,
        heuristic_fn=lambda p, s, m: heuristic.batched_distance(p, s),
        cost_weight=cost_weight,
        is_backward=True,
        inverse_action_map=inverse_action_map,
    )

    # 2. Build Loop
    init_loop_state, loop_condition, loop_body = unified_bi_search_loop_builder(
        puzzle,
        fwd_expansion,
        bwd_expansion,
        batch_size,
        max_nodes,
        pop_ratio,
        min_pop=1,
        pq_val_type=Current,
        terminate_on_first_solution=terminate_on_first_solution,
    )

    def bi_astar(
        solve_config: Puzzle.SolveConfig,
        start: Puzzle.State,
        **kwargs: Any,
    ) -> BiDirectionalSearchResult:
        """
        bi_astar is the implementation of the Bidirectional A* algorithm (Unified).
        """
        # Prepare heuristics
        # Forward Heuristic: h(state, goal)
        heuristic_params = heuristic.prepare_heuristic_parameters(solve_config, **kwargs)

        # Backward Heuristic: h(state, start)
        # We need to construct an "inverse solve config" where goal is start.
        # puzzle.get_inverse_solve_config usually does this.
        # But heuristic preparation needs valid "goal" in config.
        # We need to pass `start` as goal for backward search.

        # If puzzle supports generating inverse config:
        # inverse_solve_config = puzzle.get_inverse_solve_config(solve_config, start)
        # We assume `puzzle` has this method or we manually construct.
        # `puxle` Puzzle usually supports `make_goal_from_state`?
        # Or `solve_config` holds goal.

        # We'll assume caller passes `inverse_solve_config` or we construct it.
        # Existing `bi_astar.py` logic:
        # inverse_solve_config = puzzle.change_goal_state(solve_config, start)
        # inverse_heuristic_params = heuristic.prepare(inverse_solve_config)

        # Let's assume puzzle.get_solve_config(start) works if we want start as goal?
        # Or `solve_config` is immutable?
        # Usually `solve_config` is a dataclass.
        # We can construct new one.
        # `puzzle.traverse(start, ...)` ?

        # Let's check `bi_astar.py` original code.
        # `bi_astar_builder` in original code assumed `solve_config`.
        # Inside `bi_astar`:
        # `inverse_solve_config = puzzle.get_inverse_solve_config(solve_config, start)` (hypothetical)

        # Note: `puxle` doesn't standardize `inverse_solve_config`.
        # However, for 15-puzzle, solve_config holds target.
        # For Rubik's, usually fixed target.
        # If we reverse, we search FROM End TO Start.
        # The heuristic needs to estimate distance TO Start.
        # So we need heuristic params configured for Start.

        # Assuming `puzzle` has `get_inverse_solve_config` or similar.
        # If not, we might need a `inverse_solve_config_fn` argument?
        # But standard `bi_astar` assumed symmetric or provided mechanism.

        # We'll use `puzzle.get_inverse_solve_config` if available.
        # If not, let's assume `kwargs` has it?

        # In `bi_astar.py` from `JAxtar`:
        # `inverse_solveconfig = puzzle.get_inverse_solve_config(solve_config, start)`
        # This was present in `bi_astar.py`?
        # Let's check Step 19 summary or `bi_search_base` references.
        # `bi_search_base` has `inverse_solveconfig` in LoopState.

        # I'll use `puzzle.get_inverse_solve_config(solve_config, start)`.

        hindsight_transform = getattr(puzzle, "hindsight_transform", None)
        if callable(hindsight_transform):
            inverse_solve_config = hindsight_transform(solve_config, start)
        else:
            get_inverse_solve_config = getattr(puzzle, "get_inverse_solve_config", None)
            if callable(get_inverse_solve_config):
                inverse_solve_config = get_inverse_solve_config(solve_config, start)
            else:
                # Fallback for SlidePuzzle / Chex dataclass
                inverse_solve_config = solve_config.replace(TargetState=start)
        inverse_heuristic_params = heuristic.prepare_heuristic_parameters(
            inverse_solve_config, **kwargs
        )

        # Get Goal state from config
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
        bi_astar,
        puzzle=puzzle,
        show_compile_time=show_compile_time,
        warmup_inputs=warmup_inputs,
    )
