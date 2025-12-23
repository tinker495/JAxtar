import chex
import jax
import jax.numpy as jnp
from puxle import Sokoban

from qfunction.q_base import QFunction


class SokobanQ(QFunction):
    def __init__(self, puzzle: Sokoban):
        super().__init__(puzzle)

    def q_value(self, solve_config: Sokoban.SolveConfig, current: Sokoban.State) -> chex.Array:
        """
        Get Q values for all possible actions from the current state.
        Computes Q values as the distances between each neighboring state (generated from current)
        and the target state.
        """
        neighbors, costs = self.puzzle.get_neighbours(solve_config, current)
        dists = jax.vmap(self._distance, in_axes=(0, None))(neighbors, solve_config.TargetState)
        return dists + costs

    def _distance(self, current: Sokoban.State, target: Sokoban.State) -> float:
        """
        Compute the distance between a neighboring state and the target state for the Sokoban puzzle.
        The distance is calculated as the difference between the total number of boxes in the target state
        and the number of boxes correctly placed in the current state's corresponding positions.
        """
        # Count the total number of boxes in the target state
        target_board = target.board_unpacked
        target_box_count = jnp.sum(target_board == Sokoban.Object.BOX.value)

        # Count the number of boxes in the same position in both current and target
        current_board = current.board_unpacked
        matching_boxes = jnp.sum(
            jnp.logical_and(
                current_board == Sokoban.Object.BOX.value, target_board == Sokoban.Object.BOX.value
            )
        )

        # The heuristic value is the boxes missing from their correct positions
        return (target_box_count - matching_boxes) * 5
