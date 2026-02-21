import jax
import jax.numpy as jnp
from puxle import DotKnot

from heuristic.dotknot_utils import get_color_distances
from qfunction.q_base import QFunction


class DotKnotQ(QFunction):
    def __init__(self, puzzle: DotKnot):
        super().__init__(puzzle)

    def q_value(self, solve_config: "DotKnot.SolveConfig", current: DotKnot.State) -> float:
        """
        Get q values for all possible actions from current state.
        """
        neighbors, costs = self.puzzle.get_neighbours(solve_config, current)
        dists = jax.vmap(self._distance, in_axes=(0, None))(neighbors, solve_config.TargetState)
        return dists + costs

    def _distance(self, current: DotKnot.State) -> float:
        """
        Get distance for solving puzzle.
        """
        color_distances = jax.vmap(
            lambda color_idx: get_color_distances(
                current, color_idx, self.puzzle.size, self.puzzle.color_num
            )
        )(jnp.arange(self.puzzle.color_num))
        return jnp.sum(color_distances)
