import chex
import jax

from heuristic.neuralheuristic.neuralheuristic_base import NeuralHeuristicBase
from neural_util.modules import DTYPE
from puzzle.slidepuzzle import SlidePuzzle


class SlidePuzzleNeuralHeuristic(NeuralHeuristicBase):
    base_xy: chex.Array  # The coordinates of the numbers in the puzzle

    def __init__(self, puzzle: SlidePuzzle, init_params: bool = True):
        super().__init__(puzzle, init_params=init_params)

    def pre_process_state(self, state: SlidePuzzle.State) -> chex.Array:
        board = jax.nn.one_hot(
            state.board, num_classes=self.puzzle.size * self.puzzle.size, dtype=DTYPE
        ).flatten()
        return ((board - 0.5) * 2.0).astype(DTYPE)
