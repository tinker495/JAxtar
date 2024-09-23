import chex
import jax.numpy as jnp

from puzzle.maze import Maze

class MazeHeuristic:
    puzzle: Maze

    def __init__(self, puzzle: Maze):
        self.puzzle = puzzle

    def distance(self, current: Maze.State, target: Maze.State) -> float:
        """
        Get distance between current state and target state.
        """
        return jnp.sum(jnp.abs(current.pos.astype(int) - target.pos.astype(int)))