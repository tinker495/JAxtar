import chex
import jax
import jax.numpy as jnp

from qfunction.q_base import QFunction
from puzzle.lightsout import LightsOut

class LightsOutQ(QFunction):
    def __init__(self, puzzle: LightsOut):
        super().__init__(puzzle)

    def q_value(self, current: LightsOut.State, target: LightsOut.State) -> float:
        """
        Get q values for all possible actions from current state.
        """
        neighbors, _ = self.puzzle.get_neighbours(current)
        dists = jax.vmap(self._distance, in_axes=(0, None))(neighbors, target)
        return dists

    def _distance(self, current: LightsOut.State, target: LightsOut.State) -> float:
        """
        Get distance between current state and target state.
        """
        neq_state = jnp.not_equal(self.puzzle.from_uint8(current.board), self.puzzle.from_uint8(target.board))
        sum_neq_state = jnp.sum(neq_state)
        return sum_neq_state / 5