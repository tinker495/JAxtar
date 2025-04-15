import chex
import jax
import jax.numpy as jnp
from flax import linen as nn

from heuristic.neuralheuristic.neuralheuristic_base import NeuralHeuristicBase
from neural_util.modules import DTYPE, BatchNorm, ConvResBlock, ResBlock
from puzzle.slidepuzzle import SlidePuzzle


class SlidePuzzleNeuralHeuristic(NeuralHeuristicBase):
    def __init__(self, puzzle: SlidePuzzle, init_params: bool = True):
        self.size_square = puzzle.size * puzzle.size
        super().__init__(puzzle, init_params=init_params)

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
        # Create a one-hot encoding of the current state
        # Shape will be [puzzle_size, puzzle_size, puzzle_size*puzzle_size]
        current_board = current.board
        current_board_one_hot = jax.nn.one_hot(current_board, self.size_square).flatten()

        if self.is_fixed:
            one_hots = current_board_one_hot
        else:
            target_board = solve_config.TargetState.board
            target_board_one_hot = jax.nn.one_hot(target_board, self.size_square).flatten()
            one_hots = jnp.concatenate([target_board_one_hot, current_board_one_hot], axis=-1)

        return ((one_hots - 0.5) * 2.0).astype(DTYPE)


class Model(nn.Module):
    @nn.compact
    def __call__(self, x, training=False):
        # [4, 4, 1] -> conv
        x = nn.Conv(256, (3, 3), strides=1, padding="SAME", dtype=DTYPE)(x)
        x = BatchNorm(x, training)
        x = nn.relu(x)
        x = ConvResBlock(256, (3, 3), strides=1)(x, training)
        x = jnp.reshape(x, (x.shape[0], -1))
        x = nn.Dense(512, dtype=DTYPE)(x)
        x = BatchNorm(x, training)
        x = nn.relu(x)
        x = ResBlock(512)(x, training)
        x = nn.Dense(1, dtype=DTYPE)(x)
        return x


class SlidePuzzleConvNeuralHeuristic(NeuralHeuristicBase):
    base_xy: chex.Array  # The coordinates of the numbers in the puzzle

    def __init__(self, puzzle: SlidePuzzle, init_params: bool = True):
        x = jnp.tile(jnp.arange(puzzle.size)[:, jnp.newaxis, jnp.newaxis], (1, puzzle.size, 1))
        y = jnp.tile(jnp.arange(puzzle.size)[jnp.newaxis, :, jnp.newaxis], (puzzle.size, 1, 1))
        self.base_xy = jnp.stack([x, y], axis=2).reshape(-1, 2)
        super().__init__(puzzle, model=Model, init_params=init_params)

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
