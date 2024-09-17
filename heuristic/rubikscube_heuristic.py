import chex
import jax.numpy as jnp

from puzzle.rubikscube import RubiksCube

class RubiksCubeHeuristic:
    puzzle: RubiksCube

    def __init__(self, puzzle: RubiksCube):
        self.puzzle = puzzle

    def distance(self, current: RubiksCube.State, target: RubiksCube.State) -> float:
        """
        Get distance between current state and target state.
        """
        return 0