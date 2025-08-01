from abc import ABC, abstractmethod

import chex
import jax
from puxle import Puzzle


class QFunction(ABC):
    puzzle: Puzzle  # The puzzle rule object

    def __init__(self, puzzle: Puzzle):
        self.puzzle = puzzle

    @abstractmethod
    def q_value(self, solve_config: Puzzle.SolveConfig, current: Puzzle.State) -> chex.Array:
        """
        Get q value of the current state and target state.

        Args:
            solve_config: The solve config.
            current: The current state.

        Returns:
            The q value of the current state and target state.
            shape : (batch_size, action_size)
        """
        pass

    def batched_q_value(
        self, solve_config: Puzzle.SolveConfig, current: Puzzle.State
    ) -> chex.Array:
        """
        Get q value of the current state and target state.
        """
        return jax.vmap(self.q_value, in_axes=(None, 0))(solve_config, current)
