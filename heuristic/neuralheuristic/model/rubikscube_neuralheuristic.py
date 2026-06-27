from heuristic.neuralheuristic.neuralheuristic_base import NeuralHeuristicBase
from neural_util.basemodel import HLGResMLPModel
from neural_util.model_preprocessing import (
    RubiksCubePreProcessMixin,
    RubiksCubeRandomPreProcessMixin,
)


class RubiksCubeNeuralHeuristic(RubiksCubePreProcessMixin, NeuralHeuristicBase):
    is_fixed: bool = True


class RubiksCubeRandomNeuralHeuristic(RubiksCubeRandomPreProcessMixin, NeuralHeuristicBase):
    is_fixed: bool = False


class RubiksCubeHLGNeuralHeuristic(RubiksCubeNeuralHeuristic):
    def __init__(self, puzzle, **kwargs):
        super().__init__(puzzle, model=HLGResMLPModel, **kwargs)


class RubiksCubeRandomHLGNeuralHeuristic(RubiksCubeRandomNeuralHeuristic):
    def __init__(self, puzzle, **kwargs):
        super().__init__(puzzle, model=HLGResMLPModel, **kwargs)
