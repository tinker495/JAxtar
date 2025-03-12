import chex
import flax.linen as nn
import jax
import jax.numpy as jnp

from puzzle.sokoban import Sokoban
from qfunction.neuralq.neuralq_base import NeuralQFunctionBase


def BatchNorm(x, training):
    return nn.BatchNorm(momentum=0.9)(x, use_running_average=not training)


class SokobanTransformerModel(nn.Module):
    action_size: int
    channels: int = 16

    @nn.compact
    def __call__(self, x: chex.Array, training: bool = True) -> chex.Array:
        x = nn.Conv(self.channels // 2, (1, 1), strides=(1, 1))(
            x
        )  # (10, 10, 128) projecting the board to 128 channels
        # Add positional embedding
        batch_size, height, width, _ = x.shape

        # Create position indices
        y_pos = jnp.arange(height)[None, :, None, None]  # (1, height, 1, 1)
        x_pos = jnp.arange(width)[None, None, :, None]  # (1, 1, width, 1)

        # Broadcast to match input shape - ensure same dimensions
        y_pos = jnp.broadcast_to(y_pos, (batch_size, height, width, 1))
        x_pos = jnp.broadcast_to(x_pos, (batch_size, height, width, 1))

        # Normalize positions to be between 0 and 1
        y_pos = y_pos / height
        x_pos = x_pos / width

        # Concatenate position information with features
        pos_encoding = jnp.concatenate([y_pos, x_pos], axis=-1)

        # Project position encoding to match feature dimensions
        pos_features = nn.Dense(self.channels // 2)(pos_encoding)  # (10, 10, channels // 2)

        # Add positional embedding to input features
        x = jnp.concatenate([x, pos_features], axis=-1)  # (10, 10, channels)

        # Continue with transformer layers
        x = BatchNorm(x, training)

        # Reshape for transformer processing
        x = jnp.reshape(
            x, (batch_size, height * width, self.channels)
        )  # (batch_size, 100, channels)

        # Transformer encoder blocks
        for _ in range(3):
            # Self-attention block
            residual = x
            x = BatchNorm(x, training)
            x = nn.SelfAttention(num_heads=8)(x)
            x = residual + x

            # MLP block
            residual = x
            x = BatchNorm(x, training)
            x = nn.Dense(features=self.channels * 4)(x)
            x = nn.gelu(x)
            x = nn.Dense(features=self.channels)(x)
            x = residual + x

        # Reshape back to spatial representation
        x = jnp.reshape(x, (batch_size, -1))  # (batch_size, 100 * channels)
        x = nn.Dense(self.action_size)(x)  # (batch_size, action_size)
        return x


class SokobanNeuralQ(NeuralQFunctionBase):
    base_xy: chex.Array  # The coordinates of the numbers in the puzzle

    def __init__(self, puzzle: Sokoban, init_params: bool = True):
        super().__init__(puzzle, model=SokobanTransformerModel, init_params=init_params)

    def pre_process(self, solve_config: Sokoban.SolveConfig, current: Sokoban.State) -> chex.Array:
        target_board = self.puzzle.unpack_board(solve_config.TargetState.board)
        current_board = self.puzzle.unpack_board(current.board)
        target_board = jnp.reshape(target_board, (self.puzzle.size, self.puzzle.size, 1))
        current_board = jnp.reshape(current_board, (self.puzzle.size, self.puzzle.size, 1))
        stacked_board = jnp.concatenate([current_board, target_board], axis=-1)  # (10, 10, 2)
        one_hot_board = jax.nn.one_hot(stacked_board, num_classes=4)  # (10, 10, 2, 4)
        stacking_board = jnp.reshape(
            one_hot_board, (one_hot_board.shape[0], one_hot_board.shape[1], -1)
        )  # (10, 10, 8)
        return stacking_board
