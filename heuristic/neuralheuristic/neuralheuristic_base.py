from typing import Any

import chex
from puxle import Puzzle

from heuristic.heuristic_base import Heuristic
from neural_util.neural_distance_base import NeuralDistanceBase


class NeuralHeuristicBase(NeuralDistanceBase, Heuristic):
    load_error_name = "NeuralHeuristic"

    def prepare_heuristic_parameters(
        self, solve_config: Puzzle.SolveConfig, **kwargs: Any
    ) -> tuple[Any, Puzzle.SolveConfig]:
        return (self._params_from_kwargs(**kwargs), solve_config)

    def batched_distance(
        self,
        heuristic_parameters: tuple[Any, Puzzle.SolveConfig],
        current: Puzzle.State,
    ) -> chex.Array:
        params, solve_config = heuristic_parameters
        return self.batched_param_distance(params, solve_config, current)

    def batched_param_distance(
        self, params, solve_config: Puzzle.SolveConfig, current: Puzzle.State
    ) -> chex.Array:
        return self._batched_model_values(params, solve_config, current)

    def distance(
        self,
        heuristic_parameters: tuple[Any, Puzzle.SolveConfig],
        current: Puzzle.State,
    ) -> float:
        params, solve_config = heuristic_parameters
        return float(self.param_distance(params, solve_config, current)[0])

    def param_distance(
        self, params, solve_config: Puzzle.SolveConfig, current: Puzzle.State
    ) -> chex.Array:
        return self._model_values(params, solve_config, current)

    def post_process(self, x: chex.Array) -> chex.Array:
        return x.squeeze(1)
