import chex
import jax.numpy as jnp
from flax import linen as nn

from puzzle.slidepuzzle import SlidePuzzle
from qfunction.neuralq.neuralq_base import NeuralQFunctionBase

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


class ResBlock(nn.Module):
    node_size: int

    @nn.compact
    def __call__(self, x0):
        x = nn.LayerNorm()(x0)
        x = nn.Dense(self.node_size)(x0)
        x = nn.relu(x)
        x = nn.Dense(self.node_size)(x)
        return x + x0


class Model(nn.Module):
    action_size: int = 4

    @nn.compact
    def __call__(self, x):
        # [4, 4, 1] -> conv
        x = nn.Conv(32, (1, 1))(x)
        x = ConvResBlock(32, (3, 3), strides=1)(x)
        x = ConvResBlock(32, (3, 3), strides=1)(x)
        x = ConvResBlock(32, (3, 3), strides=1)(x)
        x = jnp.reshape(x, (x.shape[0], -1))
        x = nn.Dense(128)(x)
        x = ResBlock(128)(x)
        x = ResBlock(128)(x)
        x = nn.LayerNorm()(x)
        x = nn.Dense(self.action_size)(x)
        return x


class SlidePuzzleNeuralHeuristic(NeuralQFunctionBase):
    base_xy: chex.Array  # The coordinates of the numbers in the puzzle

    def __init__(self, puzzle: SlidePuzzle, init_params: bool = True):
        x = jnp.tile(jnp.arange(puzzle.size)[:, jnp.newaxis, jnp.newaxis], (1, puzzle.size, 1))
        y = jnp.tile(jnp.arange(puzzle.size)[jnp.newaxis, :, jnp.newaxis], (puzzle.size, 1, 1))
        self.base_xy = jnp.stack([x, y], axis=2).reshape(-1, 2)
        super().__init__(puzzle, model=Model, init_params=init_params)

    def pre_process(self, current: SlidePuzzle.State, target: SlidePuzzle.State) -> chex.Array:
        diff = self.to_2d(self._diff_pos(current, target))  # [n, n, 2]
        c_zero = self.to_2d(self._zero_pos(current))  # [n, n, 1]
        t_zero = self.to_2d(self._zero_pos(target))  # [n, n, 1]
        x = jnp.concatenate([diff, c_zero, t_zero], axis=-1)  # [n, n, 4]
        x = jnp.expand_dims(x, axis=0)
        return x

    def to_2d(self, x: chex.Array) -> chex.Array:
        return x.reshape((self.puzzle.size, self.puzzle.size, x.shape[-1]))

    def _diff_pos(self, current: SlidePuzzle.State, target: SlidePuzzle.State) -> chex.Array:
        """
        This function should return the difference between the state and the target.
        """

        def to_xy(index):
            return index // self.puzzle.size, index % self.puzzle.size

        def pos(num, board):
            return to_xy(jnp.argmax(board == num))

        tpos = jnp.array([pos(i, target.board) for i in current.board], dtype=jnp.int8)  # [16, 2]
        diff = self.base_xy - tpos  # [16, 2]
        return diff

    def _zero_pos(self, current: SlidePuzzle.State) -> chex.Array:
        """
        This function should return the zero position in the state.
        """
        return jnp.expand_dims(current.board == 0, axis=-1).astype(jnp.float32)