import chex

from heuristic.neuralheuristic.neuralheuristic_base import NeuralHeuristicBase
from neural_util.modules import DTYPE
from puzzle.lightsout import LightsOut


class LightsOutNeuralHeuristic(NeuralHeuristicBase):
    def __init__(self, puzzle: LightsOut, init_params: bool = True):
        super().__init__(puzzle, init_params=init_params)

    def pre_process_state(self, state: LightsOut.State) -> chex.Array:
        current_map = self.puzzle.from_uint8(state.board).astype(DTYPE)
        return ((current_map - 0.5) * 2.0).astype(DTYPE)
