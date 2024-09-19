import chex
import jax
import jax.numpy as jnp

from flax import linen as nn
from heuristic.DAVI.neuralheuristic_base import NeuralHeuristicBase
from puzzle.lightsout import LightsOut

NODE_SIZE = 256

class Model(nn.Module):

    @nn.compact
    def __call__(self, x):
        # [4, 4, 1] -> conv
        x = nn.Conv(512, (3, 3), padding='SAME')(x)
        x = nn.relu(x)
        x = nn.Conv(512, (3, 3))(x)
        x = nn.relu(x)
        x = jnp.reshape(x, (x.shape[0], -1))
        x = nn.Dense(512)(x)
        x = nn.relu(x)
        x = nn.Dense(256)(x)
        x = nn.relu(x)
        x = nn.Dense(1)(x)
        return x

class LightsOutNeuralHeuristic(NeuralHeuristicBase):

    def __init__(self, puzzle: LightsOut):
        super().__init__(puzzle, model=Model())

    def pre_process(self, current: LightsOut.State, target: LightsOut.State) -> chex.Array:
        x = self.to_2d(self._diff(current, target))
        x = jnp.expand_dims(x, axis=0)
        return x
    
    def to_2d(self, x: chex.Array) -> chex.Array:
        return jnp.reshape(x, (self.puzzle.size, self.puzzle.size, -1))
    
    def _diff(self, current: LightsOut.State, target: LightsOut.State) -> chex.Array:
        """
        This function should return the difference, not_equal of the current state and the target state
        """
        return jnp.not_equal(current.board, target.board)