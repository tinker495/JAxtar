import jax.numpy as jnp
from puxle import RubiksCube

from heuristic.heuristic_base import Heuristic


class RubiksCubeHeuristic(Heuristic):
    heur_modify: float

    def __init__(self, puzzle: RubiksCube):
        super().__init__(puzzle)
        if puzzle.size % 2 == 0:
            self.heur_modify = 0
        else:
            self.heur_modify = 1 / (puzzle.size**2)
        self._tile_count = puzzle.size * puzzle.size
        self._use_color_embedding = getattr(puzzle, "color_embedding", True)

    def distance(self, solve_config: RubiksCube.SolveConfig, current: RubiksCube.State) -> float:
        """
        Get distance between current state and target state.
        """
        current_faces = current.unpacked.faces
        target_faces = solve_config.TargetState.unpacked.faces
        if not self._use_color_embedding:
            current_faces = jnp.floor_divide(current_faces, self._tile_count)
            target_faces = jnp.floor_divide(target_faces, self._tile_count)
        equal_faces = 1 - (jnp.equal(current_faces, target_faces).mean(1) - self.heur_modify) / (
            1 - self.heur_modify
        )
        # center of faces are not considered
        return equal_faces.sum() * 2
