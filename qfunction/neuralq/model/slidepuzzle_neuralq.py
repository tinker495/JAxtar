from neural_util.conv_models import SlidePuzzleConvModel
from neural_util.model_preprocessing import (
    SlidePuzzleConvPreProcessMixin,
    SlidePuzzlePreProcessMixin,
)
from qfunction.neuralq.neuralq_base import NeuralQFunctionBase


class SlidePuzzleNeuralQ(SlidePuzzlePreProcessMixin, NeuralQFunctionBase):
    is_fixed: bool = True


class SlidePuzzleRandomNeuralQ(SlidePuzzleNeuralQ):
    is_fixed: bool = False


class SlidePuzzleConvNeuralQ(SlidePuzzleConvPreProcessMixin, NeuralQFunctionBase):
    network_model = SlidePuzzleConvModel
