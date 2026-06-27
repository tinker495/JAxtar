from neural_util.model_preprocessing import PancakePreProcessMixin
from qfunction.neuralq.neuralq_base import NeuralQFunctionBase


class PancakeNeuralQ(PancakePreProcessMixin, NeuralQFunctionBase):
    is_fixed: bool = True


class PancakeRandomNeuralQ(PancakeNeuralQ):
    is_fixed: bool = False
