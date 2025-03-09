import chex
import jax.numpy as jnp

from puzzle.world_model import WorldModelPuzzleBase
from qfunction.neuralq.neuralq_base import NeuralQFunctionBase


class WorldModelNeuralQ(NeuralQFunctionBase):
    def __init__(self, puzzle: WorldModelPuzzleBase, init_params: bool = True):
        super().__init__(puzzle, init_params=init_params)

    def solve_config_pre_process(
        self, solve_config: WorldModelPuzzleBase.SolveConfig
    ) -> chex.Array:
        target_latent = self.puzzle.from_uint8(solve_config.TargetState.latent).astype(jnp.float32)
        return (target_latent - 0.5) * 2.0

    def state_pre_process(self, state: WorldModelPuzzleBase.State) -> chex.Array:
        current_latent = self.puzzle.from_uint8(state.latent).astype(jnp.float32)
        return (current_latent - 0.5) * 2.0
