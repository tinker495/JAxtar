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
        current_faces = current.faces
        target_faces = target.faces
        equal_faces = 1 - jnp.equal(current_faces, target_faces).mean(1)
        return equal_faces.sum()