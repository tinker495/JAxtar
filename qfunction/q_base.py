from abc import ABC, abstractmethod

import chex
import jax

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

    def batched_q_value(self, current: Puzzle.State, target: Puzzle.State) -> chex.Array:
        """
        Get q value of the current state and target state.
        """
        return jax.vmap(self.q_value, in_axes=(0, None))(current, target)


class EmptyQFunction(QFunction):
    def q_value(self, current: Puzzle.State, target: Puzzle.State) -> chex.Array:
        neighbors, _ = self.puzzle.get_neighbours(current)
        dists = jax.vmap(self._distance, in_axes=(0, None))(neighbors, target)
        return dists

    def _distance(self, current: Puzzle.State, target: Puzzle.State) -> float:
        return 0
