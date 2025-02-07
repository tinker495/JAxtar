import chex
import jax.numpy as jnp

from qfunction.q_base import QFunction


class EmptyQFunction(QFunction):
    def __init__(self, puzzle):
        super().__init__(puzzle)

    def q_value(self, current: chex.Array, target: chex.Array) -> chex.Array:
        """
        Get q values for all possible actions from the current state.
        For EmptyQFunction, this returns an array of zeros for each available move.
        """
        neighbors, _ = self.puzzle.get_neighbours(current)
        return jnp.zeros(neighbors.shape[0])
