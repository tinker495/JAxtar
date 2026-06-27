from heuristic.neuralheuristic.neuralheuristic_base import NeuralHeuristicBase
from neural_util.conv_models import SlidePuzzleConvModel
from neural_util.model_preprocessing import (
    SlidePuzzleConvPreProcessMixin,
    SlidePuzzlePreProcessMixin,
)


class SlidePuzzleNeuralHeuristic(SlidePuzzlePreProcessMixin, NeuralHeuristicBase):
    is_fixed: bool = True


class SlidePuzzleRandomNeuralHeuristic(SlidePuzzleNeuralHeuristic):
    is_fixed: bool = False


class SlidePuzzleConvNeuralHeuristic(SlidePuzzleConvPreProcessMixin, NeuralHeuristicBase):
    network_model = SlidePuzzleConvModel
