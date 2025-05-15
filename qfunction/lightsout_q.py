import jax
import jax.numpy as jnp

from puzzle import LightsOut, from_uint8
from qfunction.q_base import QFunction


class LightsOutQ(QFunction):
    def __init__(self, puzzle: LightsOut):
        super().__init__(puzzle)

    def q_value(self, solve_config: LightsOut.SolveConfig, current: LightsOut.State) -> float:
        """
        Get q values for all possible actions from current state.
        """
        neighbors, _ = self.puzzle.get_neighbours(solve_config, current)
        dists = jax.vmap(self._distance, in_axes=(0, None))(neighbors, solve_config.TargetState)
        return dists

    def _distance(self, current: LightsOut.State, target: LightsOut.State) -> float:
        """
        Get distance between current state and target state.
        """
        neq_state = jnp.not_equal(
            from_uint8(current.board, (self.puzzle.size * self.puzzle.size)),
            from_uint8(target.board, (self.puzzle.size * self.puzzle.size)),
        )
        sum_neq_state = jnp.sum(neq_state)
        return sum_neq_state / 5 * 2.0
