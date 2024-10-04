import chex
import jax.numpy as jnp

from heuristic.heuristic_base import Heuristic
from puzzle.maze import Maze

class MazeHeuristic(Heuristic):
    def __init__(self, puzzle: Maze):
        super().__init__(puzzle)

    def distance(self, current: Maze.State, target: Maze.State) -> float:
        """
        Get distance between current state and target state.
        """
        return jnp.sum(jnp.abs(current.pos.astype(int) - target.pos.astype(int)))