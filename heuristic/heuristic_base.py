from abc import ABC, abstractmethod
from typing import Any, Union

import chex
import jax
from puxle import Puzzle


class Heuristic(ABC):
    puzzle: Puzzle  # The puzzle rule object
    is_fixed: bool = (
        False  # True if this heuristic is only valid for a fixed target (i.e. it does not
    )
    # support arbitrary solve_config targets / retargeting).

    def __init__(self, puzzle: Puzzle):
        self.puzzle = puzzle

    def prepare_heuristic_parameters(
        self, solve_config: Puzzle.SolveConfig, **kwargs: Any
    ) -> Union[Puzzle.SolveConfig, Any]:
        """
        This function prepares the parameters for use in distance calculations.
        By default, it returns the solve config unchanged. Subclasses can override
        this to transform the solve config into a more efficient representation,
        such as embedding it into a special vector (e.g., via a neural network)
        to guide the search toward the goal. During the search process, this method
        is called once at the beginning to prepare parameters that remain invariant
        throughout the search iterations.

        For example, this enables cases where the solve config is converted to
        an embedding vector for processing in specialized neural networks.

        If a neural net heuristic or similar case, this method's output may include
        the neural net's parameters or similar.

        Returns:
            The prepared parameters, which could be the original or a transformed representation.
        """
        return solve_config

    def batched_distance(
        self, heuristic_parameters: Union[Puzzle.SolveConfig, Any], current: Puzzle.State
    ) -> chex.Array:
        """
        This function should return the distance between the state and the target.

        Args:
            heuristic_parameters: The parameters for the heuristic. normally a SolveConfig
            current: The current state.

        Returns:
            The distance between the state and the target.
            shape : (batch_size,)
        """
        return jax.vmap(self.distance, in_axes=(None, 0))(heuristic_parameters, current)

    @abstractmethod
    def distance(
        self, heuristic_parameters: Union[Puzzle.SolveConfig, Any], current: Puzzle.State
    ) -> float:
        """
        This function should return the distance between the state and the target.

        Args:
            heuristic_parameters: The parameters for the heuristic. normally a SolveConfig
            current: The current state.

        Returns:
            The distance between the state and the target.
            shape : single scalar
        """
        pass
