import chex
import jax.numpy as jnp
from flax import linen as nn
from puxle import SlidePuzzle

from neural_util.basemodel import DistanceModel
from neural_util.dtypes import DTYPE, PARAM_DTYPE
from neural_util.modules import DEFAULT_NORM_FN, ConvResBlock, ResBlock, apply_norm
from neural_util.preprocessing import (
    slidepuzzle_diff_pos,
    slidepuzzle_pre_process,
    slidepuzzle_zero_pos,
)
from qfunction.neuralq.neuralq_base import NeuralQFunctionBase


class SlidePuzzleNeuralQ(NeuralQFunctionBase):
    is_fixed: bool = True

    def __init__(self, puzzle: SlidePuzzle, **kwargs):
        self.size_square = puzzle.size * puzzle.size
        super().__init__(puzzle, **kwargs)

    def pre_process(
        self, solve_config: SlidePuzzle.SolveConfig, current: SlidePuzzle.State
    ) -> chex.Array:
        """
        Pre-process the current state for the neural heuristic model.

        Args:
            solve_config: Configuration for solving the puzzle
            current: Current state of the puzzle

        Returns:
            One-hot representation of the puzzle state
        """
        target_board = solve_config.TargetState.board_unpacked
        return slidepuzzle_pre_process(
            current.board_unpacked, target_board, self.size_square, self.is_fixed
        )


class SlidePuzzleRandomNeuralQ(SlidePuzzleNeuralQ):
    is_fixed: bool = False


class Model(DistanceModel):
    norm_fn: callable = DEFAULT_NORM_FN

    @nn.compact
    def __call__(self, x, training=False):
        # [4, 4, 1] -> conv
        x = nn.Conv(256, (3, 3), strides=1, padding="SAME", dtype=DTYPE, param_dtype=PARAM_DTYPE)(x)
        x = apply_norm(self.norm_fn, x, training)
        x = nn.relu(x)
        x = ConvResBlock(256, (3, 3), strides=1, norm_fn=self.norm_fn)(x, training)
        x = jnp.reshape(x, (x.shape[0], -1))
        x = nn.Dense(512, dtype=DTYPE, param_dtype=PARAM_DTYPE)(x)
        x = apply_norm(self.norm_fn, x, training)
        x = nn.relu(x)
        x = ResBlock(512, norm_fn=self.norm_fn)(x, training)
        x = nn.Dense(self.action_size, dtype=DTYPE, param_dtype=PARAM_DTYPE)(x)
        return x


class SlidePuzzleConvNeuralQ(NeuralQFunctionBase):
    base_xy: chex.Array  # The coordinates of the numbers in the puzzle

    def __init__(self, puzzle: SlidePuzzle, **kwargs):
        x = jnp.tile(jnp.arange(puzzle.size)[:, jnp.newaxis, jnp.newaxis], (1, puzzle.size, 1))
        y = jnp.tile(jnp.arange(puzzle.size)[jnp.newaxis, :, jnp.newaxis], (puzzle.size, 1, 1))
        self.base_xy = jnp.stack([x, y], axis=2).reshape(-1, 2)
        super().__init__(puzzle, model=Model, **kwargs)

    def pre_process(
        self, solve_config: SlidePuzzle.SolveConfig, current: SlidePuzzle.State
    ) -> chex.Array:
        diff = self.to_2d(self._diff_pos(current, solve_config.TargetState))  # [n, n, 2]
        c_zero = self.to_2d(self._zero_pos(current))  # [n, n, 1]
        t_zero = self.to_2d(self._zero_pos(solve_config.TargetState))  # [n, n, 1]
        x = jnp.concatenate([diff, c_zero, t_zero], axis=-1)  # [n, n, 4]
        return x.astype(DTYPE)

    def to_2d(self, x: chex.Array) -> chex.Array:
        return x.reshape((self.puzzle.size, self.puzzle.size, x.shape[-1]))

    def _diff_pos(self, current: SlidePuzzle.State, target: SlidePuzzle.State) -> chex.Array:
        """Return the per-tile position difference between the current and target states."""
        return slidepuzzle_diff_pos(
            current.board_unpacked, target.board_unpacked, self.base_xy, self.puzzle.size
        )

    def _zero_pos(self, current: SlidePuzzle.State) -> chex.Array:
        """Return a binary mask indicating the blank-tile position."""
        return slidepuzzle_zero_pos(current.board_unpacked)
