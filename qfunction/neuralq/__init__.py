from __future__ import annotations

from _lazy_imports import lazy_dir, load_lazy_export

__all__ = [
    "LightsOutConvNeuralQ",
    "LightsOutNeuralQ",
    "LightsOutRandomNeuralQ",
    "PancakeNeuralQ",
    "PancakeRandomNeuralQ",
    "RubiksCubeHLGNeuralQ",
    "RubiksCubeNeuralQ",
    "RubiksCubeRandomHLGNeuralQ",
    "RubiksCubeRandomNeuralQ",
    "SlidePuzzleConvNeuralQ",
    "SlidePuzzleNeuralQ",
    "SlidePuzzleRandomNeuralQ",
    "SokobanNeuralQ",
    "WorldModelNeuralQ",
    "NeuralQFunctionBase",
]

_EXPORTS = {
    "LightsOutConvNeuralQ": "qfunction.neuralq.model.lightsout_neuralq",
    "LightsOutNeuralQ": "qfunction.neuralq.model.lightsout_neuralq",
    "LightsOutRandomNeuralQ": "qfunction.neuralq.model.lightsout_neuralq",
    "PancakeNeuralQ": "qfunction.neuralq.model.pancake_neuralq",
    "PancakeRandomNeuralQ": "qfunction.neuralq.model.pancake_neuralq",
    "RubiksCubeHLGNeuralQ": "qfunction.neuralq.model.rubikscube_neuralq",
    "RubiksCubeNeuralQ": "qfunction.neuralq.model.rubikscube_neuralq",
    "RubiksCubeRandomHLGNeuralQ": "qfunction.neuralq.model.rubikscube_neuralq",
    "RubiksCubeRandomNeuralQ": "qfunction.neuralq.model.rubikscube_neuralq",
    "SlidePuzzleConvNeuralQ": "qfunction.neuralq.model.slidepuzzle_neuralq",
    "SlidePuzzleNeuralQ": "qfunction.neuralq.model.slidepuzzle_neuralq",
    "SlidePuzzleRandomNeuralQ": "qfunction.neuralq.model.slidepuzzle_neuralq",
    "SokobanNeuralQ": "qfunction.neuralq.model.sokoban_neuralq",
    "WorldModelNeuralQ": "qfunction.neuralq.model.world_model_neuralq",
    "NeuralQFunctionBase": "qfunction.neuralq.neuralq_base",
}


def __getattr__(name: str):
    return load_lazy_export(name, __name__, _EXPORTS, globals())


def __dir__() -> list[str]:
    return lazy_dir(globals(), __all__)
