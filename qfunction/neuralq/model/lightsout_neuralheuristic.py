import chex
import jax.numpy as jnp
from flax import linen as nn

from heuristic.neuralheuristic.neuralheuristic_base import NeuralHeuristicBase
from puzzle.lightsout import LightsOut

NODE_SIZE = 256


class ConvResBlock(nn.Module):
    filters: int
    kernel_size: int
    strides: int

    @nn.compact
    def __call__(self, x0):
        x = nn.LayerNorm()(x0)
        x = nn.Conv(self.filters, self.kernel_size, strides=self.strides, padding="SAME")(x)
        x = nn.relu(x)
        x = nn.Conv(self.filters, self.kernel_size, strides=self.strides, padding="SAME")(x)
        return x + x0


class Model(nn.Module):
    action_size: int

    @nn.compact
    def __call__(self, x):
        # [4, 4, 1] -> conv
        x = (x - 0.5) * 2.0
        x = nn.Conv(32, (1, 1))(x)
        x = ConvResBlock(32, (3, 3), strides=1)(x)
        x = ConvResBlock(32, (3, 3), strides=1)(x)
        x = ConvResBlock(32, (3, 3), strides=1)(x)
        x = ConvResBlock(32, (3, 3), strides=1)(x)
        x = ConvResBlock(32, (3, 3), strides=1)(x)
        x = nn.LayerNorm()(x)
        x = jnp.reshape(x, (x.shape[0], -1))
        chex.assert_shape(x, (None, self.action_size))
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
        return jnp.not_equal(current_map, target_map).astype(jnp.float32)
