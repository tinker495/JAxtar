from typing import Any, Optional

import jax.numpy as jnp

from heuristic.heuristic_base import Heuristic
from puzzle.sokoban import Object, Sokoban


class SokobanHeuristic(Heuristic):
    def __init__(self, puzzle: Sokoban):
        super().__init__(puzzle)

    def distance(
        self,
        solve_config: Sokoban.SolveConfig,
        current: Sokoban.State,
        params: Optional[Any] = None,
    ) -> float:
        """
        Simple heuristic for the Sokoban puzzle.
        It computes the distance as the total number of boxes in the target state
        minus the number of boxes that are correctly placed (i.e., matching positions in current and target).
        Assumes that boxes are represented by 1 in both current and target arrays.
        """
        # Count the total number of boxes in the target state
        target_board = self.puzzle.unpack_board(solve_config.TargetState.board)
        target_box_count = jnp.sum(target_board == Object.BOX.value)

        # Count the number of boxes in the same position in both current and target
        current_board = self.puzzle.unpack_board(current.board)
        matching_boxes = jnp.sum(
            jnp.logical_and(current_board == Object.BOX.value, target_board == Object.BOX.value)
        )

        # The heuristic value is the boxes missing from their correct positions
        return (target_box_count - matching_boxes) * 5
