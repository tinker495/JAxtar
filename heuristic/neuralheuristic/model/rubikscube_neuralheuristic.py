from heuristic.neuralheuristic.neuralheuristic_base import NeuralHeuristicBase
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


class RubiksCubeTransformerNeuralHeuristic(RubiksCubeTokenPreProcessMixin, NeuralHeuristicBase):
    is_fixed: bool = True

    def __init__(self, puzzle, **kwargs):
        super().__init__(puzzle, model=DiscreteVarTransformerModel, **kwargs)


class RubiksCubeRandomTransformerNeuralHeuristic(
    RubiksCubeRandomTokenPreProcessMixin, NeuralHeuristicBase
):
    is_fixed: bool = False

    def __init__(self, puzzle, **kwargs):
        super().__init__(puzzle, model=DiscreteVarTransformerModel, **kwargs)


class RubiksCubeTransformerHybridNeuralHeuristic(
    RubiksCubeTokenPreProcessMixin, NeuralHeuristicBase
):
    is_fixed: bool = True

    def __init__(self, puzzle, **kwargs):
        super().__init__(puzzle, model=DiscreteVarTransformerHybridModel, **kwargs)


class RubiksCubeRandomTransformerHybridNeuralHeuristic(
    RubiksCubeRandomTokenPreProcessMixin, NeuralHeuristicBase
):
    is_fixed: bool = False

    def __init__(self, puzzle, **kwargs):
        super().__init__(puzzle, model=DiscreteVarTransformerHybridModel, **kwargs)
