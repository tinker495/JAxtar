from abc import ABC, abstractmethod

import chex

from puzzle.puzzle_base import Puzzle


class QFunction(ABC):
    puzzle: Puzzle  # The puzzle rule object

    def __init__(self, puzzle: Puzzle):
        self.puzzle = puzzle

    @abstractmethod
    def q_value(self, current: Puzzle.State, target: Puzzle.State) -> chex.Array:
        """
        Get q value of the current state and target state.

        Args:
            current: The current state.
            target: The target state.

        Returns:
            The q value of the current state and target state.
            shape : (batch_size, action_size)
        """
        pass
