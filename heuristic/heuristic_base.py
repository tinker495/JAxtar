from abc import ABC, abstractmethod

import chex
import jax
from puxle import Puzzle


class Heuristic(ABC):
    puzzle: Puzzle  # The puzzle rule object

    def __init__(self, puzzle: Puzzle):
        self.puzzle = puzzle

    def batched_distance(
        self, solve_config: Puzzle.SolveConfig, current: Puzzle.State
    ) -> chex.Array:
        """
        This function should return the distance between the state and the target.
        """
        return jax.vmap(self.distance, in_axes=(None, 0))(solve_config, current)

    @abstractmethod
    def distance(self, solve_config: Puzzle.SolveConfig, current: Puzzle.State) -> float:
        """
        This function should return the distance between the state and the target.

        Args:
            solve_config: The solve config.
            current: The current state.

        Returns:
            The distance between the state and the target.
            shape : single scalar
        """
        pass
