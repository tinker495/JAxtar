import jax.numpy as jnp
from puxle import Maze

from heuristic.heuristic_base import Heuristic


class MazeHeuristic(Heuristic):
    def __init__(self, puzzle: Maze):
        super().__init__(puzzle)

    def distance(self, solve_config: Maze.SolveConfig, current: Maze.State) -> float:
        """
        Get distance between current state and target state.
        """
        return jnp.sum(jnp.abs(current.pos.astype(int) - solve_config.TargetState.pos.astype(int)))
