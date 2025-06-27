import chex
import jax
import jax.numpy as jnp
from puxle import Sokoban

from neural_util.modules import DTYPE
from qfunction.neuralq.neuralq_base import NeuralQFunctionBase


class SokobanNeuralQ(NeuralQFunctionBase):
    base_xy: chex.Array  # The coordinates of the numbers in the puzzle

    def __init__(self, puzzle: Sokoban, **kwargs):
        super().__init__(puzzle, **kwargs)

    def pre_process(self, solve_config: Sokoban.SolveConfig, current: Sokoban.State) -> chex.Array:
        target_board = solve_config.TargetState.unpacked.board
        current_board = current.unpacked.board
        stacked_board = jnp.concatenate([current_board, target_board], axis=-1)
        one_hot_board = jax.nn.one_hot(stacked_board, num_classes=4)
        flattened_board = jnp.reshape(one_hot_board, (-1,))
        return ((flattened_board - 0.5) * 2.0).astype(DTYPE)
