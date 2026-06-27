from neural_util.basemodel import HLGResMLPModel
from neural_util.model_preprocessing import (
    RubiksCubePreProcessMixin,
    RubiksCubeRandomPreProcessMixin,
)
from qfunction.neuralq.neuralq_base import NeuralQFunctionBase


class RubiksCubeNeuralQ(RubiksCubePreProcessMixin, NeuralQFunctionBase):
    is_fixed: bool = True


class RubiksCubeRandomNeuralQ(RubiksCubeRandomPreProcessMixin, NeuralQFunctionBase):
    is_fixed: bool = False


class RubiksCubeHLGNeuralQ(RubiksCubeNeuralQ):
    def __init__(self, puzzle, **kwargs):
        super().__init__(puzzle, model=HLGResMLPModel, **kwargs)


class RubiksCubeRandomHLGNeuralQ(RubiksCubeRandomNeuralQ):
    def __init__(self, puzzle, **kwargs):
        super().__init__(puzzle, model=HLGResMLPModel, **kwargs)
