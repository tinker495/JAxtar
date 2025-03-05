import chex
import jax.numpy as jnp
from flax import linen as nn

from heuristic.neuralheuristic.neuralheuristic_base import NeuralHeuristicBase
from puzzle.world_model import WorldModelPuzzleBase

NODE_SIZE = 256

PROJECTION_DIM = 250


class Projector(nn.Module):
    @nn.compact
    def __call__(self, x, training=False):
        x = nn.Dense(1000)(x)
        x = nn.BatchNorm(use_running_average=not training)(x)
        x = nn.relu(x)
        x = nn.Dense(1000)(x)
        x = nn.relu(x)
        x = nn.Dense(PROJECTION_DIM)(x)
        return x


class Model(nn.Module):
    @nn.compact
    def __call__(self, latent_stack, training=False):
        current_projection = Projector()(latent_stack[..., 0], training=training)
        target_projection = Projector()(latent_stack[..., 1], training=training)
        scalar = jnp.sum(current_projection * target_projection, axis=-1, keepdims=True)
        return scalar


class WorldModelNeuralHeuristic(NeuralHeuristicBase):
    base_xy: chex.Array  # The coordinates of the numbers in the puzzle

    def __init__(self, puzzle: WorldModelPuzzleBase, init_params: bool = True):
        super().__init__(puzzle, Model(), init_params=init_params)

    def pre_process(
        self, solve_config: WorldModelPuzzleBase.SolveConfig, current: WorldModelPuzzleBase.State
    ) -> chex.Array:
        target_latent = self.puzzle.representation_solve_config(solve_config)
        current_latent = self.puzzle.representation_state(current)
        stack_latent = jnp.stack([current_latent, target_latent], axis=-1)
        return stack_latent
