import jax
import jax.numpy as jnp
from puxle import DotKnot

from heuristic.dotknot_utils import get_color_distances
from heuristic.heuristic_base import Heuristic
from puxle.core.puzzle_base import EmptyInstanceContext
from puxle.puzzles.dotknot import DotKnotGoalSpec, DotKnotSolveConfig, DotKnotState


class DotKnotHeuristic(Heuristic[DotKnotState, EmptyInstanceContext, DotKnotGoalSpec]):
    def __init__(self, puzzle: DotKnot):
        super().__init__(puzzle)

    def distance(self, solve_config: DotKnotSolveConfig, current: DotKnotState) -> jax.Array:
        """
        Get distance for solving puzzle.
        """
        color_distances = jax.vmap(
            lambda color_idx: get_color_distances(
                current, color_idx, self.puzzle.size, self.puzzle.color_num
            )
        )(jnp.arange(self.puzzle.color_num))
        return jnp.sum(color_distances).astype(jnp.float32)
