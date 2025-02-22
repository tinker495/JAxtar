import chex
import jax.numpy as jnp

from puzzle.world_model import WorldModelPuzzleBase
from qfunction.neuralq.neuralq_base import NeuralQFunctionBase

NODE_SIZE = 256


class WorldModelNeuralQ(NeuralQFunctionBase):
    base_xy: chex.Array  # The coordinates of the numbers in the puzzle

    def __init__(self, puzzle: WorldModelPuzzleBase, init_params: bool = True):
        super().__init__(puzzle, init_params=init_params)

    def pre_process(
        self, solve_config: WorldModelPuzzleBase.SolveConfig, current: WorldModelPuzzleBase.State
    ) -> chex.Array:
        target_latent = self.puzzle.from_uint8(solve_config.TargetState.latent).astype(jnp.float32)
        current_latent = self.puzzle.from_uint8(current.latent).astype(jnp.float32)
        latent_diff = target_latent - current_latent
        return latent_diff
