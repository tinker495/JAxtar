from abc import ABC, abstractmethod

from puzzle.puzzle_base import Puzzle


class Heuristic(ABC):
    puzzle: Puzzle  # The puzzle rule object

    def __init__(self, puzzle: Puzzle):
        self.puzzle = puzzle

    @abstractmethod
    def distance(self, current: Puzzle.State, target: Puzzle.State) -> float:
        """
        This function should return the distance between the state and the target.

        Args:
            current: The current state.
            target: The target state.

        Returns:
            The distance between the state and the target.
            shape : single scalar
        """
        pass
