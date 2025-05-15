from typing import Any, Optional

import chex
import jax
import jax.numpy as jnp

from puzzle import TSP
from qfunction.q_base import QFunction


class TSPQ(QFunction):
    def __init__(self, puzzle):
        super().__init__(puzzle)

    def q_value(
        self, solve_config: TSP.SolveConfig, current: TSP.State, params: Optional[Any] = None
    ) -> chex.Array:
        """
        Get q values for all possible actions from the current state.
        For EmptyQFunction, this returns an array of zeros for each available move.
        """
        neighbors, _ = self.puzzle.get_neighbours(solve_config, current)
        dists = jax.vmap(self._distance, in_axes=(None, 0))(solve_config, neighbors)
        return dists

    def _distance(self, solve_config: TSP.SolveConfig, current: TSP.State) -> float:
        """
        Return zero distance for any puzzle state.
        """
        inv_mask = 1 - current.unpacking().mask
        distance_matrix = solve_config.distance_matrix
        masked_dists = distance_matrix * inv_mask[None, :] * inv_mask[:, None]
        total_cost = jnp.mean(jnp.sum(masked_dists, axis=1))
        return total_cost
