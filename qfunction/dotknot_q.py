import jax
import jax.numpy as jnp

from puzzle.dotknot import DotKnot
from qfunction.q_base import QFunction


class DotKnotQ(QFunction):
    def __init__(self, puzzle: DotKnot):
        super().__init__(puzzle)

    def q_value(self, current: DotKnot.State, target: DotKnot.State) -> float:
        """
        Get q values for all possible actions from current state.
        """
        neighbors, _ = self.puzzle.get_neighbours(current)
        dists = jax.vmap(self._distance, in_axes=(0, None))(neighbors, target)
        return dists

    def _distance(self, current: DotKnot.State, target: DotKnot.State) -> float:
        """
        Get distance for solving puzzle.
        """
        color_distances = jax.vmap(self.get_color_distances, in_axes=(None, 0))(
            current, jnp.arange(self.puzzle.color_num)
        )
        return jnp.sum(color_distances)

    def get_color_distances(self, current: DotKnot.State, color_idx: int) -> float:
        """
        Get distance for solving puzzle.
        """
        point_a = current.board == (color_idx + 1)
        point_a_available = jnp.any(point_a)
        point_a_pos = jnp.stack(
            jnp.unravel_index(jnp.argmax(point_a), (self.puzzle.size, self.puzzle.size))
        )
        point_b = current.board == (color_idx + self.puzzle.color_num + 1)
        point_b_pos = jnp.stack(
            jnp.unravel_index(jnp.argmax(point_b), (self.puzzle.size, self.puzzle.size))
        )
        return jnp.where(point_a_available, jnp.sum(jnp.abs(point_a_pos - point_b_pos)), 0)