import chex
import jax.numpy as jnp

from heuristic.neuralheuristic.neuralheuristic_base import NeuralHeuristicBase
from puzzle.world_model import WorldModelPuzzleBase


class WorldModelNeuralHeuristic(NeuralHeuristicBase):
    base_xy: chex.Array  # The coordinates of the numbers in the puzzle

    def __init__(self, puzzle: WorldModelPuzzleBase, init_params: bool = True):
        super().__init__(puzzle, init_params=init_params)

    def pre_process(
        self, solve_config: WorldModelPuzzleBase.SolveConfig, current: WorldModelPuzzleBase.State
    ) -> chex.Array:
        target_latent = self.puzzle.from_uint8(solve_config.TargetState.latent).astype(jnp.float32)
        current_latent = self.puzzle.from_uint8(current.latent).astype(jnp.float32)
        latent_stack = jnp.concatenate([current_latent, target_latent], axis=-1)
        latent_stack = jnp.reshape(latent_stack, (-1,))
        return (latent_stack - 0.5) * 2.0
