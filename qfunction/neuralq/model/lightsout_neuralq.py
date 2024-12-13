import chex
import jax.numpy as jnp
from flax import linen as nn

from puzzle.lightsout import LightsOut
from qfunction.neuralq.neuralq_base import NeuralQFunctionBase

NODE_SIZE = 256


class ConvResBlock(nn.Module):
    filters: int
    kernel_size: int
    strides: int

    @nn.compact
    def __call__(self, x0, training: bool = True):
        x = nn.Conv(self.filters, self.kernel_size, strides=self.strides, padding="SAME")(x0)
        x = nn.BatchNorm()(x, use_running_average=not training)
        x = nn.relu(x)
        x = nn.Conv(self.filters, self.kernel_size, strides=self.strides, padding="SAME")(x)
        x = nn.BatchNorm()(x, use_running_average=not training)
        return nn.relu(x + x0)


class Model(nn.Module):
    action_size: int

    @nn.compact
    def __call__(self, x, training: bool = True):
        # [4, 4, 1] -> conv
        x = nn.Conv(256, (3, 3), strides=1, padding="SAME")(x)
        x = nn.BatchNorm()(x, use_running_average=not training)
        x = nn.relu(x)
        x = nn.Conv(32, (3, 3), strides=1, padding="SAME")(x)
        x = nn.BatchNorm()(x, use_running_average=not training)
        x = nn.relu(x)
        x = ConvResBlock(32, (3, 3), strides=1)(x, training)
        x = ConvResBlock(32, (3, 3), strides=1)(x, training)
        value_feature = jnp.reshape(x, (x.shape[0], -1))
        value = nn.Dense(1)(value_feature)
        adv = nn.Conv(1, (1, 1), strides=1, padding="SAME")(x)
        adv = jnp.reshape(adv, (x.shape[0], -1))
        q = value + adv - jnp.mean(adv, axis=1, keepdims=True)
        chex.assert_shape(q, (None, self.action_size))
        return q


class LightsOutNeuralQ(NeuralQFunctionBase):
    def __init__(self, puzzle: LightsOut, init_params: bool = True):
        super().__init__(puzzle, model=Model, init_params=init_params)

    def pre_process(self, current: LightsOut.State, target: LightsOut.State) -> chex.Array:
        x = self.to_2d(self._diff(current, target))
        return (x - 0.5) * 2.0

    def to_2d(self, x: chex.Array) -> chex.Array:
        return jnp.reshape(x, (self.puzzle.size, self.puzzle.size, 1))

    def _diff(self, current: LightsOut.State, target: LightsOut.State) -> chex.Array:
        """
        This function should return the difference, not_equal of the current state and the target state
        """
        current_map = self.puzzle.from_uint8(current.board)
        target_map = self.puzzle.from_uint8(target.board)
        return jnp.not_equal(current_map, target_map).astype(jnp.float32)
