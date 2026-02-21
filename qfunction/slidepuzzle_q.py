import chex
import jax
import jax.numpy as jnp
from puxle import SlidePuzzle

from heuristic.slidepuzzle_utils import diff_pos, linear_conflict
from qfunction.q_base import QFunction


class SlidePuzzleQ(QFunction):
    base_xy: chex.Array  # The coordinates of the numbers in the puzzle

    def __init__(self, puzzle: SlidePuzzle):
        super().__init__(puzzle)
        x = jnp.tile(
            jnp.arange(self.puzzle.size)[:, jnp.newaxis, jnp.newaxis], (1, self.puzzle.size, 1)
        )
        y = jnp.tile(
            jnp.arange(self.puzzle.size)[jnp.newaxis, :, jnp.newaxis], (self.puzzle.size, 1, 1)
        )
        self.base_xy = jnp.stack([x, y], axis=2).reshape(-1, 2)

    def q_value(self, solve_config: SlidePuzzle.SolveConfig, current: SlidePuzzle.State) -> float:
        """
        This function should return the q value of the current state and target state.
        """
        neighbors, costs = self.puzzle.get_neighbours(solve_config, current)
        dists = jax.vmap(self._distance, in_axes=(0, None))(neighbors, solve_config.TargetState)
        return dists + costs

    def _distance(self, current: SlidePuzzle.State, target: SlidePuzzle.State) -> float:
        """
        This function should return the distance between the state and the target.
        """
        d, tpos = diff_pos(current, target, self.puzzle.size, self.base_xy)
        not_empty = current.board_unpacked != 0
        return (
            self._manhattan_distance(not_empty, d)
            + linear_conflict(tpos, not_empty, d, self.puzzle.size)
        ).astype(jnp.float32)

    def _manhattan_distance(self, not_empty, diff) -> int:
        """
        This function should return the manhattan distance between the state and the target.
        """
        return jnp.sum(not_empty * jnp.sum(jnp.abs(diff), axis=1))
