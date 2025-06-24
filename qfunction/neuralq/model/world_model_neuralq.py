import chex
import jax.numpy as jnp

from neural_util.modules import DTYPE
from qfunction.neuralq.neuralq_base import NeuralQFunctionBase
from world_model_puzzle import WorldModelPuzzleBase


class WorldModelNeuralQ(NeuralQFunctionBase):
    base_xy: chex.Array  # The coordinates of the numbers in the puzzle

    def __init__(self, puzzle: WorldModelPuzzleBase, **kwargs):
        super().__init__(puzzle, **kwargs)

    def pre_process(
        self, solve_config: WorldModelPuzzleBase.SolveConfig, current: WorldModelPuzzleBase.State
    ) -> chex.Array:
        target_latent = solve_config.TargetState.unpacking().latent.astype(jnp.float32)
        current_latent = current.unpacking().latent.astype(jnp.float32)
        latent_stack = jnp.concatenate([current_latent, target_latent], axis=-1)
        latent_stack = jnp.reshape(latent_stack, (-1,))
        return ((latent_stack - 0.5) * 2.0).astype(DTYPE)
