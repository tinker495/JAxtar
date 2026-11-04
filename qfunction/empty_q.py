import chex
import jax.numpy as jnp

from qfunction.q_base import QFunction
from puxle.core.puzzle_base import SolveConfigView
from xtructure import Xtructurable


class EmptyQFunction(QFunction):
    def __init__(self, puzzle):
        super().__init__(puzzle)

    def q_value(self, solve_config: SolveConfigView, current: Xtructurable) -> chex.Array:
        """
        Get q values for all possible actions from the current state.
        For EmptyQFunction, this returns an array of zeros for each available move.
        """
        _, next_costs = self.puzzle.get_neighbours(solve_config, current)
        return jnp.zeros_like(next_costs)
