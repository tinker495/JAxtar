import jax
import jax.numpy as jnp
from puxle import Maze

from heuristic.heuristic_base import Heuristic
from puxle.puzzles.maze import MazeInstanceContext, MazeSolveConfig, MazeState


class MazeHeuristic(Heuristic[MazeState, MazeInstanceContext, MazeState]):
    def __init__(self, puzzle: Maze):
        super().__init__(puzzle)

    def distance(self, solve_config: MazeSolveConfig, current: MazeState) -> jax.Array:
        """
        Get distance between current state and target state.
        """
        return jnp.sum(
            jnp.abs(current.pos.astype(int) - solve_config.GoalSpec.pos.astype(int))
        ).astype(jnp.float32)
