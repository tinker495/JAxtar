import chex
import jax.numpy as jnp
from puxle import SlidePuzzle

from heuristic.heuristic_base import Heuristic
from heuristic.slidepuzzle_utils import diff_pos, linear_conflict


class SlidePuzzleHeuristic(Heuristic):
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

    def distance(self, solve_config: SlidePuzzle.SolveConfig, current: SlidePuzzle.State) -> float:
        """
        This function should return the distance between the state and the target.
        """
        diff, tpos = diff_pos(current, solve_config.TargetState, self.puzzle.size, self.base_xy)
        not_empty = current.board_unpacked != 0
        return (
            jnp.sum(not_empty * jnp.sum(jnp.abs(diff), axis=1))
            + linear_conflict(tpos, not_empty, diff, self.puzzle.size)
        ).astype(jnp.float32)
