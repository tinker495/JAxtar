from typing import Any, Optional

import jax
import jax.numpy as jnp
from puxle import Maze

from qfunction.q_base import QFunction


class MazeQ(QFunction):
    def __init__(self, puzzle: Maze):
        super().__init__(puzzle)

    def q_value(
        self, solve_config: Maze.SolveConfig, current: Maze.State, params: Optional[Any] = None
    ) -> float:
        """
        Get q values for all possible actions from current state.
        """
        neighbors, costs = self.puzzle.get_neighbours(solve_config, current)
        dists = jax.vmap(self._distance, in_axes=(None, 0))(solve_config, neighbors)
        return dists + costs

    def _distance(self, solve_config: Maze.SolveConfig, current: Maze.State) -> float:
        """
        Get distance between current state and target state.
        """
        return jnp.sum(jnp.abs(current.pos.astype(int) - solve_config.TargetState.pos.astype(int)))
