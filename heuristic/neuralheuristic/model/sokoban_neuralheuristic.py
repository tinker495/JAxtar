import chex
import jax
import jax.numpy as jnp

from heuristic.neuralheuristic.neuralheuristic_base import NeuralHeuristicBase
from puzzle.sokoban import Object, Sokoban


class SokobanNeuralHeuristic(NeuralHeuristicBase):
    base_xy: chex.Array  # The coordinates of the numbers in the puzzle

    def __init__(self, puzzle: Sokoban, init_params: bool = True):
        super().__init__(puzzle, init_params=init_params)

    def pre_process(self, solve_config: Sokoban.SolveConfig, current: Sokoban.State) -> chex.Array:
        target_board = self.puzzle.unpack_board(solve_config.TargetState.board)
        target_board = target_board == Object.BOX.value
        current_board = self.puzzle.unpack_board(current.board)
        current_one_hot = jax.nn.one_hot(current_board, num_classes=4)
        stacked_board = jnp.concatenate([current_one_hot, target_board[:, jnp.newaxis]], axis=-1)
        flattened_board = jnp.reshape(stacked_board, (-1,))
        return (flattened_board - 0.5) * 2.0
