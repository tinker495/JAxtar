import chex
import jax.numpy as jnp
from flax import linen as nn

from puzzle.lightsout import LightsOut
from qfunction.neuralq.neuralq_base import NeuralQFunctionBase

NODE_SIZE = 256


def BatchNorm(x, training):
    return nn.BatchNorm(momentum=0.9)(x, use_running_average=not training)


class ConvResBlock(nn.Module):
    filters: int
    kernel_size: int
    strides: int

    @nn.compact
    def __call__(self, x0, training=False):
        x = nn.Conv(self.filters, self.kernel_size, strides=self.strides, padding="SAME")(x0)
        x = BatchNorm(x, training)
        x = nn.relu(x)
        x = nn.Conv(self.filters, self.kernel_size, strides=self.strides, padding="SAME")(x)
        x = BatchNorm(x, training)
        return nn.relu(x + x0)


class ResBlock(nn.Module):
    node_size: int

    @nn.compact
    def __call__(self, x0, training=False):
        x = nn.Dense(self.node_size)(x0)
        x = BatchNorm(x, training)
        x = nn.relu(x)
        x = nn.Dense(self.node_size)(x)
        x = BatchNorm(x, training)
        return nn.relu(x + x0)


class Model(nn.Module):
    action_size: int

    @nn.compact
    def __call__(self, x, training=False):
        # [4, 4, 1] -> conv
        x = nn.Conv(64, (3, 3), strides=1, padding="SAME")(x)
        x = BatchNorm(x, training)
        x = nn.relu(x)
        x = ConvResBlock(64, (3, 3), strides=1)(x, training)
        x = jnp.reshape(x, (x.shape[0], -1))
        x = nn.Dense(1024)(x)
        x = BatchNorm(x, training)
        x = nn.relu(x)
        x = ResBlock(1024)(x, training)
        x = nn.Dense(self.action_size)(x)
        return x


class LightsOutNeuralQ(NeuralQFunctionBase):
    def __init__(self, puzzle: LightsOut, init_params: bool = True):
        super().__init__(puzzle, model=Model, init_params=init_params)

    def pre_process(
        self, solve_config: LightsOut.SolveConfig, current: LightsOut.State
    ) -> chex.Array:
        x = self.to_2d(self._diff(current, solve_config.TargetState))
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
