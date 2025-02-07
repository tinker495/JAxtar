import chex
import jax
import jax.numpy as jnp

from puzzle.sokoban import Object, Sokoban
from qfunction.q_base import QFunction


class SokobanQ(QFunction):
    def __init__(self, puzzle: Sokoban):
        super().__init__(puzzle)

    def q_value(self, current: Sokoban.State, target: Sokoban.State) -> chex.Array:
        """
        Get Q values for all possible actions from the current state.
        Computes Q values as the distances between each neighboring state (generated from current)
        and the target state.
        """
        neighbors, _ = self.puzzle.get_neighbours(current)
        return jax.vmap(self._distance, in_axes=(0, None))(neighbors, target)

    def _distance(self, current: Sokoban.State, target: Sokoban.State) -> float:
        """
        Compute the distance between a neighboring state and the target state for the Sokoban puzzle.
        The distance is calculated as the difference between the total number of boxes in the target state
        and the number of boxes correctly placed in the current state's corresponding positions.
        """
        # Count the total number of boxes in the target state
        target_board = Sokoban.unpack_board(target.board)
        target_box_count = jnp.sum(target_board == Object.BOX.value)

        # Count the number of boxes in the same position in both current and target
        current_board = Sokoban.unpack_board(current.board)
        matching_boxes = jnp.sum(
            jnp.logical_and(current_board == Object.BOX.value, target_board == Object.BOX.value)
        )

        # The heuristic value is the boxes missing from their correct positions
        return (target_box_count - matching_boxes) * 5
