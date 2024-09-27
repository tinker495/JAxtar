import chex
import jax
import jax.numpy as jnp

from flax import linen as nn
from heuristic.DAVI.neuralheuristic_base import NeuralHeuristicBase
from puzzle.lightsout import LightsOut

NODE_SIZE = 256

class ConvResBlock(nn.Module):
    filters: int
    kernel_size: int
    strides: int

    @nn.compact
    def __call__(self, x):
        x0 = nn.Conv(self.filters, 1)(x) # 1x1 conv to pass information through
        x = nn.Conv(self.filters, self.kernel_size, strides=self.strides, padding='SAME')(x)
        x = nn.relu(x)
        x = nn.Conv(self.filters, self.kernel_size, strides=self.strides, padding='SAME')(x)
        x = nn.relu(x)
        x = x + x0
        return x

class Model(nn.Module):

    @nn.compact
    def __call__(self, x):
        # [4, 4, 1] -> conv
        x = ConvResBlock(128, (3, 3), strides=1)(x)
        x = ConvResBlock(128, (3, 3), strides=1)(x)
        x = ConvResBlock(128, (3, 3), strides=1)(x)
        x = jnp.reshape(x, (x.shape[0], -1))
        x = nn.Dense(256)(x)
        x = nn.relu(x)
        x = nn.Dense(128)(x)
        x = nn.relu(x)
        x = nn.Dense(1)(x)
        return x

class LightsOutNeuralHeuristic(NeuralHeuristicBase):

    def __init__(self, puzzle: LightsOut, init_params: bool = True):
        super().__init__(puzzle, model=Model(), init_params=init_params)

    def pre_process(self, current: LightsOut.State, target: LightsOut.State) -> chex.Array:
        x = self.to_2d(self._diff(current, target))
        x = jnp.expand_dims(x, axis=0)
        return x
    
    def to_2d(self, x: chex.Array) -> chex.Array:
        return jnp.reshape(x, (self.puzzle.size, self.puzzle.size, 1))
    
    def _diff(self, current: LightsOut.State, target: LightsOut.State) -> chex.Array:
        """
        This function should return the difference, not_equal of the current state and the target state
        """
        current_map = self.puzzle.from_uint8(current.board)
        target_map = self.puzzle.from_uint8(target.board)
        return jnp.not_equal(current_map, target_map).astype(jnp.float32) - 0.5