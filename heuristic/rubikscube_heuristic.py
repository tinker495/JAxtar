import jax
import jax.numpy as jnp
from puxle import RubiksCube

from heuristic.heuristic_base import Heuristic
from puxle.core.puzzle_base import EmptyInstanceContext
from puxle.puzzles.rubikscube import RubiksCubeSolveConfig, RubiksCubeState


class RubiksCubeHeuristic(Heuristic[RubiksCubeState, EmptyInstanceContext, RubiksCubeState]):
    heur_modify: float

    def __init__(self, puzzle: RubiksCube):
        super().__init__(puzzle)
        if puzzle.size % 2 == 0:
            self.heur_modify = 0
        else:
            self.heur_modify = 1 / (puzzle.size**2)
        self._tile_count = puzzle.size * puzzle.size
        self._use_color_embedding = getattr(puzzle, "color_embedding", True)

    def distance(self, solve_config: RubiksCubeSolveConfig, current: RubiksCubeState) -> jax.Array:
        """
        Get distance between current state and target state.
        """
        current_faces = current.faces_unpacked
        target_faces = solve_config.GoalSpec.faces_unpacked
        if not self._use_color_embedding:
            current_faces = jnp.floor_divide(current_faces, self._tile_count)
            target_faces = jnp.floor_divide(target_faces, self._tile_count)
        equal_faces = 1 - (jnp.equal(current_faces, target_faces).mean(1) - self.heur_modify) / (
            1 - self.heur_modify
        )
        # center of faces are not considered
        return equal_faces.sum() * 2
