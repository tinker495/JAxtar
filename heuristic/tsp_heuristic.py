import jax.numpy as jnp

from heuristic.heuristic_base import Heuristic
from puzzle.tsp import TSP


class TSPHeuristic(Heuristic):
    def __init__(self, puzzle):
        super().__init__(puzzle)

    def distance(self, solve_config: TSP.SolveConfig, current: TSP.State) -> float:
        """
        Return zero distance for any puzzle state.
        """
        inv_mask = 1 - self.puzzle.from_uint8(current.mask)
        distance_matrix = solve_config.distance_matrix
        masked_dists = distance_matrix * inv_mask[None, :] * inv_mask[:, None]
        total_cost = jnp.mean(jnp.sum(masked_dists, axis=1))
        return total_cost
