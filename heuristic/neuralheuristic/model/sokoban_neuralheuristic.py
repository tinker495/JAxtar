import chex
import jax
import jax.numpy as jnp

from heuristic.neuralheuristic.neuralheuristic_base import NeuralHeuristicBase
from neural_util.modules import DTYPE
from puzzle.sokoban import Sokoban


class SokobanNeuralHeuristic(NeuralHeuristicBase):
    base_xy: chex.Array  # The coordinates of the numbers in the puzzle

    def __init__(self, puzzle: Sokoban, init_params: bool = True):
        super().__init__(puzzle, init_params=init_params)

    def pre_process_state(self, state: Sokoban.State) -> chex.Array:
        board = self.puzzle.unpack_board(state.board)
        one_hot_board = jax.nn.one_hot(board, num_classes=4)
        flattened_board = jnp.reshape(one_hot_board, (-1,))
        return ((flattened_board - 0.5) * 2.0).astype(DTYPE)
