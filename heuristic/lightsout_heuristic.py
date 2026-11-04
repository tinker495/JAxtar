import jax
import jax.numpy as jnp
from puxle import LightsOut

from heuristic.heuristic_base import Heuristic
from puxle.core.puzzle_base import EmptyInstanceContext
from puxle.puzzles.lightsout import LightsOutSolveConfig, LightsOutState


class LightsOutHeuristic(Heuristic[LightsOutState, EmptyInstanceContext, LightsOutState]):
    def __init__(self, puzzle: LightsOut):
        super().__init__(puzzle)

    def distance(self, solve_config: LightsOutSolveConfig, current: LightsOutState) -> jax.Array:
        """
        Get distance between current state and target state.
        """
        neq_state = jnp.not_equal(
            current.board_unpacked,
            solve_config.GoalSpec.board_unpacked,
        )
        sum_neq_state = jnp.sum(neq_state)
        return sum_neq_state / 5 * 2
