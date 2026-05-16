from __future__ import annotations

import importlib
from typing import Any

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


def __getattr__(name: str) -> Any:
    try:
        module_name = _EXPORTS[name]
    except KeyError as exc:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from exc
    value = getattr(importlib.import_module(module_name), name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))
