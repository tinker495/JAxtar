from typing import Any

import jax
import jax.numpy as jnp
from puxle import Puzzle

from helpers.jax_compile import jit_with_warmup
from JAxtar.annotate import MIN_BATCH_SIZE
from JAxtar.core.common import finalize_search_result
from JAxtar.core.expansion import QStarExpansion
from JAxtar.core.loop import unified_search_loop_builder
from JAxtar.core.result import ParentWithCosts, SearchResult
from JAxtar.core.scoring import QStarScoring
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
    look_ahead_pruning: bool = True,
    pessimistic_update: bool = True,
    warmup_inputs: tuple[Puzzle.SolveConfig, Puzzle.State] | None = None,
):
    """
    Builds and returns a JAX-accelerated Q* search function using Unified Core.
    """
    # 0. Build Switcher
    variable_q_batch_switcher = variable_batch_switcher_builder(
        q_fn.batched_q_value,
        max_batch_size=batch_size,
        min_batch_size=MIN_BATCH_SIZE,
        pad_value=jnp.inf,
    )

    # Adapter for Q-function: (params, states, mask) -> Q(s, a) [action, batch]
    # variable_q_batch_switcher returns [batch, action] usually (if batched_q_value does).
    # But qstar.py expected [action, batch] or transposed it.
    # QStarExpansion logic seems to assume flat arrays or matching shapes.
    # If variable_q_batch_switcher returns [batch, action],
    # QStarExpansion logic needs to flatten correctly.
    # QStarExpansion uses build_action_major_parent_context which iterates action then batch?
    # No, build_action_major_parent_context usually returns [batch*action] where action is inner?
    # Actually it returns [batch, action] or similar.
    # We should ensure `heuristic_fn` returns [batch, action] and QStarExpansion FLATTENS it.
    # My QStarExpansion implementation called `self.q_fn(..., filled)` then `.flatten()`.
    # Standard flattening is Row-Major (Batch, Action).
    # If q_switcher returns (Batch, Action), then flatten is (B*A).
    # But build_action_major_parent_context usually expects (Action, Batch) logic for packing?
    # Let's check `build_action_major_parent_context`.
    # It returns `unflatten_shape`.

    # If `qstar.py` logic was `q_vals.transpose()`, it means it wanted (Action, Batch).
    # `sort_and_pack_action_candidates` expects (Action, Batch).
    # So `QStarExpansion` needs `q_vals` in (Action, Batch) or needs to handle it.

    # My QStarExpansion implementation:
    # `q_vals = self.q_fn(...)`
    # `costs` from helper are flattened.
    # Helper usually returns Action-Major if requested?
    # name is `build_action_major_parent_context`.
    # So it probably produces Action-Major arrays.
    # So q_vals must be Action-Major (Action, Batch).
    # So we MUST transpose if switcher returns (Batch, Action).

    def q_fn_adapter(params, states, mask):
        q = variable_q_batch_switcher(params, states, mask)
        return q.transpose().astype(jnp.float32)  # [Action, Batch]

    # 1. Policies
    scoring_policy = QStarScoring()

    expansion_policy = QStarExpansion(
        scoring_policy=scoring_policy,
        q_fn=q_fn_adapter,
        cost_weight=cost_weight,
        look_ahead_pruning=look_ahead_pruning,
        pessimistic_update=pessimistic_update,
    )

    # 2. Build Loop
    init_loop_state, loop_condition, loop_body = unified_search_loop_builder(
        puzzle,
        expansion_policy,
        batch_size,
        max_nodes,
        pop_ratio,
        min_pop=1,  # derived? qstar.py used derived min_pop
        pq_val_type=ParentWithCosts,
    )

    # qstar.py derived min_pop: max(1, MIN_BATCH_SIZE // (action_size//2)).
    # We can pass that if we want matching behavior.
    # Let's calculate it.
    denom = max(1, puzzle.action_size // 2)
    min_pop = max(1, MIN_BATCH_SIZE // denom)

    # Re-build with correct min_pop
    init_loop_state, loop_condition, loop_body = unified_search_loop_builder(
        puzzle,
        expansion_policy,
        batch_size,
        max_nodes,
        pop_ratio,
        min_pop=min_pop,
        pq_val_type=ParentWithCosts,
    )

    def qstar(
        solve_config: Puzzle.SolveConfig,
        start: Puzzle.State,
        **kwargs: Any,
    ) -> SearchResult:
        """
        qstar implementation (Unified).
        """
        # Prepare Q parameters
        q_parameters = q_fn.prepare_q_parameters(solve_config, **kwargs)

        loop_state = init_loop_state(solve_config, start, heuristic_params=q_parameters)

        loop_state = jax.lax.while_loop(loop_condition, loop_body, loop_state)

        search_result = loop_state.search_result
        current = loop_state.current
        states = loop_state.states
        filled = loop_state.filled

        solved_mask = jnp.logical_and(puzzle.batched_is_solved(solve_config, states), filled)
        return finalize_search_result(search_result, current, solved_mask)

    return jit_with_warmup(
        qstar,
        puzzle=puzzle,
        show_compile_time=show_compile_time,
        warmup_inputs=warmup_inputs,
    )
