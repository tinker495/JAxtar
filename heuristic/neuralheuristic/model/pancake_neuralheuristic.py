from heuristic.neuralheuristic.neuralheuristic_base import NeuralHeuristicBase
from neural_util.model_preprocessing import PancakePreProcessMixin


class PancakeNeuralHeuristic(PancakePreProcessMixin, NeuralHeuristicBase):
    is_fixed: bool = True


class PancakeRandomNeuralHeuristic(PancakeNeuralHeuristic):
    is_fixed: bool = False
