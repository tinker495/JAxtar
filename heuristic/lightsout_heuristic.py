from typing import Any, Optional

import jax.numpy as jnp

from heuristic.heuristic_base import Heuristic
from puzzle import LightsOut


class LightsOutHeuristic(Heuristic):
    def __init__(self, puzzle: LightsOut):
        super().__init__(puzzle)

    def distance(
        self,
        solve_config: LightsOut.SolveConfig,
        current: LightsOut.State,
        params: Optional[Any] = None,
    ) -> float:
        """
        Get distance between current state and target state.
        """
        neq_state = jnp.not_equal(
            current.unpacking().board,
            solve_config.TargetState.unpacking().board,
        )
        sum_neq_state = jnp.sum(neq_state)
        return sum_neq_state / 5 * 2
