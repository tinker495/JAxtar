import chex
import jax.numpy as jnp
from puxle import Puzzle

from qfunction.q_base import QFunction


class EmptyQFunction(QFunction):
    def __init__(self, puzzle):
        super().__init__(puzzle)

    def q_value(self, solve_config: Puzzle.SolveConfig, current: Puzzle.State) -> chex.Array:
        _, next_costs = self.puzzle.get_neighbours(solve_config, current)
        return jnp.zeros_like(next_costs)
