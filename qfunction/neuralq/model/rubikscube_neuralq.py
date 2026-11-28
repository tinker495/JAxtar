from neural_util.basemodel import (
    DiscreteVarTransformerHybridModel,
    DiscreteVarTransformerModel,
    HLGResMLPModel,
)
from neural_util.model_preprocessing import (
    RubiksCubePreProcessMixin,
    RubiksCubeRandomPreProcessMixin,
    RubiksCubeRandomTokenPreProcessMixin,
    RubiksCubeTokenPreProcessMixin,
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


class RubiksCubeTransformerNeuralQ(RubiksCubeTokenPreProcessMixin, NeuralQFunctionBase):
    is_fixed: bool = True

    def __init__(self, puzzle, **kwargs):
        super().__init__(puzzle, model=DiscreteVarTransformerModel, **kwargs)


class RubiksCubeRandomTransformerNeuralQ(RubiksCubeRandomTokenPreProcessMixin, NeuralQFunctionBase):
    is_fixed: bool = False

    def __init__(self, puzzle, **kwargs):
        super().__init__(puzzle, model=DiscreteVarTransformerModel, **kwargs)


class RubiksCubeTransformerHybridNeuralQ(RubiksCubeTokenPreProcessMixin, NeuralQFunctionBase):
    is_fixed: bool = True

    def __init__(self, puzzle, **kwargs):
        super().__init__(puzzle, model=DiscreteVarTransformerHybridModel, **kwargs)


class RubiksCubeRandomTransformerHybridNeuralQ(
    RubiksCubeRandomTokenPreProcessMixin, NeuralQFunctionBase
):
    is_fixed: bool = False

    def __init__(self, puzzle, **kwargs):
        super().__init__(puzzle, model=DiscreteVarTransformerHybridModel, **kwargs)
