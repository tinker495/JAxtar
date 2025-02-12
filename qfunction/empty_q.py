import chex
import jax.numpy as jnp

from puzzle.puzzle_base import Puzzle
from qfunction.q_base import QFunction


class EmptyQFunction(QFunction):
    def __init__(self, puzzle):
        super().__init__(puzzle)

    def q_value(self, solve_config: Puzzle.SolveConfig, current: Puzzle.State) -> chex.Array:
        """
        Get q values for all possible actions from the current state.
        For EmptyQFunction, this returns an array of zeros for each available move.
        """
        _, next_costs = self.puzzle.get_neighbours(solve_config, current)
        return jnp.zeros_like(next_costs)
