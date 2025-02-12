import jax
import jax.numpy as jnp

from puzzle.maze import Maze
from qfunction.q_base import QFunction


class MazeQ(QFunction):
    def __init__(self, puzzle: Maze):
        super().__init__(puzzle)

    def q_value(self, solve_config: Maze.SolveConfig, current: Maze.State) -> float:
        """
        Get q values for all possible actions from current state.
        """
        neighbors, _ = self.puzzle.get_neighbours(solve_config, current)
        dists = jax.vmap(self._distance, in_axes=(0, None))(neighbors, solve_config.TargetState)
        return dists

    def _distance(self, current: Maze.State, target: Maze.State) -> float:
        """
        Get distance between current state and target state.
        """
        return jnp.sum(jnp.abs(current.pos.astype(int) - target.pos.astype(int)))
