import chex
import jax.numpy as jnp

from puzzle.lightsout import LightsOut
from qfunction.neuralq.neuralq_base import NeuralQFunctionBase


class LightsOutNeuralQ(NeuralQFunctionBase):
    def __init__(self, puzzle: LightsOut, init_params: bool = True):
        super().__init__(puzzle, init_params=init_params)

    def solve_config_pre_process(self, solve_config: LightsOut.SolveConfig) -> chex.Array:
        map = self.puzzle.from_uint8(solve_config.TargetState.board).astype(jnp.float32)
        flattened_map = map.flatten()
        return (flattened_map - 0.5) * 2.0

    def state_pre_process(self, state: LightsOut.State) -> chex.Array:
        map = self.puzzle.from_uint8(state.board).astype(jnp.float32)
        flattened_map = map.flatten()
        return (flattened_map - 0.5) * 2.0
