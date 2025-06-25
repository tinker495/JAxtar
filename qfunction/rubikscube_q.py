import jax
import jax.numpy as jnp
from puxle import RubiksCube

from qfunction.q_base import QFunction


class RubiksCubeQ(QFunction):
    heur_modify: float

    def __init__(self, puzzle: RubiksCube):
        super().__init__(puzzle)
        if puzzle.size % 2 == 0:
            self.heur_modify = 0
        else:
            self.heur_modify = 1 / (puzzle.size**2)

    def q_value(self, solve_config: RubiksCube.SolveConfig, current: RubiksCube.State) -> float:
        """
        Get q values for all possible actions from current state.
        """
        neighbors, _ = self.puzzle.get_neighbours(solve_config, current)
        dists = jax.vmap(self._distance, in_axes=(0, None))(neighbors, solve_config.TargetState)
        return dists

    def _distance(self, current: RubiksCube.State, target: RubiksCube.State) -> float:
        """
        Get distance between current state and target state.
        """
        current_faces = current.unpacked.faces
        target_faces = target.unpacked.faces
        equal_faces = 1 - (jnp.equal(current_faces, target_faces).mean(1) - self.heur_modify) / (
            1 - self.heur_modify
        )
        # center of faces are not considered
        return equal_faces.sum() * 2
