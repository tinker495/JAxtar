import chex
import jax

from puzzle.slidepuzzle import SlidePuzzle
from qfunction.neuralq.neuralq_base import NeuralQFunctionBase


class SlidePuzzleNeuralQ(NeuralQFunctionBase):
    def __init__(self, puzzle: SlidePuzzle, init_params: bool = True):
        super().__init__(puzzle, init_params=init_params)

    def solve_config_pre_process(self, solve_config: SlidePuzzle.SolveConfig) -> chex.Array:
        board = solve_config.TargetState.board
        one_hot = jax.nn.one_hot(board, num_classes=self.puzzle.size * self.puzzle.size).flatten()
        return (one_hot - 0.5) * 2.0

    def state_pre_process(self, state: SlidePuzzle.State) -> chex.Array:
        board = state.board
        one_hot = jax.nn.one_hot(board, num_classes=self.puzzle.size * self.puzzle.size).flatten()
        return (one_hot - 0.5) * 2.0
