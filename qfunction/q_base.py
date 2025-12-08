from abc import ABC, abstractmethod
from typing import Any, Union

import chex
import jax
from puxle import Puzzle


class QFunction(ABC):
    puzzle: Puzzle  # The puzzle rule object

    def __init__(self, puzzle: Puzzle):
        self.puzzle = puzzle

    def prepare_q_parameters(
        self, solve_config: Puzzle.SolveConfig, **kwargs: Any
    ) -> Union[Puzzle.SolveConfig, Any]:
        """
        This function prepares the parameters for use in Q-value calculations.
        By default, it returns the solve config unchanged. Subclasses can override
        this to transform the solve config into a more efficient representation,
        such as embedding it into a special vector (e.g., via a neural network)
        to guide the search toward the goal. During the search process, this method
        is called once at the beginning to prepare parameters that remain invariant
        throughout the search iterations.

        For example, this enables cases where the solve config is converted to
        an embedding vector for processing in specialized neural networks.

        If a neural net Q-function or similar case, this method's output may include
        the neural net's parameters or similar.

        Returns:
            The prepared parameters, which could be the original or a transformed representation.
        """
        return solve_config

    @abstractmethod
    def q_value(
        self, q_parameters: Union[Puzzle.SolveConfig, Any], current: Puzzle.State
    ) -> chex.Array:
        """
        Get q value of the current state and target state.

        Args:
            q_parameters: The parameters for the Q-function. normally a SolveConfig
            current: The current state.

        Returns:
            The q value of the current state and target state.
            shape : (batch_size, action_size)
        """
        pass

    def batched_q_value(
        self, q_parameters: Union[Puzzle.SolveConfig, Any], current: Puzzle.State
    ) -> chex.Array:
        """
        Get q value of the current state and target state.

        Args:
            q_parameters: The parameters for the Q-function. normally a SolveConfig
            current: The current state.

        Returns:
            The q value of the current state and target state.
            shape : (batch_size, action_size)
        """
        return jax.vmap(self.q_value, in_axes=(None, 0))(q_parameters, current)
