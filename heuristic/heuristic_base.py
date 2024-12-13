from abc import ABC, abstractmethod

import chex
import jax

from puzzle.puzzle_base import Puzzle


class Heuristic(ABC):
    puzzle: Puzzle  # The puzzle rule object

    def __init__(self, puzzle: Puzzle):
        self.puzzle = puzzle

    def batched_distance(self, current: Puzzle.State, target: Puzzle.State) -> chex.Array:
        """
        This function should return the distance between the state and the target.
        """
        return jax.vmap(self.distance, in_axes=(0, None))(current, target)

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
