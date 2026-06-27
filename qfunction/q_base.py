from abc import ABC, abstractmethod
from typing import Any, Union

import chex
import jax
from puxle import Puzzle

from heuristic.heuristic_base import Heuristic


class QFunction(ABC):
    puzzle: Puzzle  # The puzzle rule object
    is_fixed: bool = (
        False  # True if this Q-function is only valid for a fixed target (i.e. it does not
    )
    # support arbitrary solve_config targets / retargeting).

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


class QFromHeuristic(QFunction):
    """Q-function adapter for hand-written heuristics.

    Most non-neural Q-functions evaluate every neighbor with the matching
    heuristic and add the transition cost. Keeping that pattern here avoids
    copying puzzle-specific distance formulas into both ``heuristic`` and
    ``qfunction`` modules.
    """

    heuristic: Heuristic

    def __init__(self, heuristic: Heuristic):
        super().__init__(heuristic.puzzle)
        self.heuristic = heuristic
        self.is_fixed = heuristic.is_fixed

    def prepare_q_parameters(
        self, solve_config: Puzzle.SolveConfig, **kwargs: Any
    ) -> tuple[Any, Puzzle.SolveConfig]:
        heuristic_parameters = self.heuristic.prepare_heuristic_parameters(solve_config, **kwargs)
        return heuristic_parameters, solve_config

    def q_value(
        self,
        q_parameters: Union[Puzzle.SolveConfig, tuple[Any, Puzzle.SolveConfig]],
        current: Puzzle.State,
    ) -> chex.Array:
        heuristic_parameters, solve_config = self._split_q_parameters(q_parameters)
        neighbors, costs = self.puzzle.get_neighbours(solve_config, current)
        return self.heuristic.batched_distance(heuristic_parameters, neighbors) + costs

    def _split_q_parameters(
        self, q_parameters: Union[Puzzle.SolveConfig, tuple[Any, Puzzle.SolveConfig]]
    ) -> tuple[Any, Puzzle.SolveConfig]:
        if isinstance(q_parameters, tuple) and len(q_parameters) == 2:
            return q_parameters
        return q_parameters, q_parameters
