import jax.numpy as jnp

from heuristic.heuristic_base import Heuristic
from puzzle.lightsout import LightsOut


class LightsOutHeuristic(Heuristic):
    def __init__(self, puzzle: LightsOut):
        super().__init__(puzzle)

    def distance(self, solve_config: LightsOut.SolveConfig, current: LightsOut.State) -> float:
        """
        Get distance between current state and target state.
        """
        neq_state = jnp.not_equal(
            self.puzzle.from_uint8(current.board),
            self.puzzle.from_uint8(solve_config.TargetState.board),
        )
        sum_neq_state = jnp.sum(neq_state)
        return sum_neq_state / 5 * 2
