from typing import Any

import jax

from heuristic.heuristic_base import Heuristic
from neural_util.neural_distance_base import NeuralDistanceBase
from puxle.core.puzzle_base import SolveConfigView
from xtructure import Xtructurable


class NeuralHeuristicBase(NeuralDistanceBase, Heuristic):
    load_error_name = "NeuralHeuristic"

    def prepare_heuristic_parameters(
        self, solve_config: SolveConfigView, **kwargs: Any
    ) -> tuple[Any, SolveConfigView]:
        return (self._params_from_kwargs(**kwargs), solve_config)

    def batched_distance(
        self,
        heuristic_parameters: tuple[Any, SolveConfigView],
        current: Xtructurable,
    ) -> jax.Array:
        params, solve_config = heuristic_parameters
        return self.batched_param_distance(params, solve_config, current)

    def batched_param_distance(
        self, params, solve_config: SolveConfigView, current: Xtructurable
    ) -> jax.Array:
        return self._batched_model_values(params, solve_config, current)

    def distance(
        self,
        heuristic_parameters: tuple[Any, SolveConfigView],
        current: Xtructurable,
    ) -> float:
        params, solve_config = heuristic_parameters
        return float(self.param_distance(params, solve_config, current)[0])

    def param_distance(
        self, params, solve_config: SolveConfigView, current: Xtructurable
    ) -> jax.Array:
        return self._model_values(params, solve_config, current)

    def post_process(self, x: jax.Array) -> jax.Array:
        return x.squeeze(1)
