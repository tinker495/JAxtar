from typing import Any

import chex
from puxle import Puzzle

from neural_util.neural_distance_base import NeuralDistanceBase
from qfunction.q_base import QFunction


class NeuralQFunctionBase(NeuralDistanceBase, QFunction):
    load_error_name = "NeuralQFunction"

    def _configure_puzzle(self, puzzle: Puzzle) -> None:
        self.action_size = puzzle.action_size

    def _model_kwargs(self, resolved_kwargs: dict[str, Any]) -> dict[str, Any]:
        return {**resolved_kwargs, "action_size": self.action_size}

    def _build_model(self, model, resolved_kwargs: dict[str, Any]):
        return model(self.action_size, **resolved_kwargs)

    def prepare_q_parameters(
        self, solve_config: Puzzle.SolveConfig, **kwargs: Any
    ) -> tuple[Any, Puzzle.SolveConfig]:
        return (self._params_from_kwargs(**kwargs), solve_config)

    def batched_q_value(
        self, q_parameters: tuple[Any, Puzzle.SolveConfig], current: Puzzle.State
    ) -> chex.Array:
        params, solve_config = q_parameters
        return self.batched_param_q_value(params, solve_config, current)

    def batched_param_q_value(
        self, params, solve_config: Puzzle.SolveConfig, current: Puzzle.State
    ) -> chex.Array:
        return self._batched_model_values(params, solve_config, current)

    def q_value(self, q_parameters: tuple[Any, Puzzle.SolveConfig], current: Puzzle.State):
        params, solve_config = q_parameters
        return self.param_q_value(params, solve_config, current)[0]

    def param_q_value(
        self, params, solve_config: Puzzle.SolveConfig, current: Puzzle.State
    ) -> chex.Array:
        return self._model_values(params, solve_config, current)
