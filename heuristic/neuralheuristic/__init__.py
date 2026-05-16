from __future__ import annotations

import importlib
from typing import Any

__all__ = [
    "LightsOutConvNeuralHeuristic",
    "LightsOutNeuralHeuristic",
    "LightsOutRandomNeuralHeuristic",
    "PancakeNeuralHeuristic",
    "PancakeRandomNeuralHeuristic",
    "RubiksCubeHLGNeuralHeuristic",
    "RubiksCubeNeuralHeuristic",
    "RubiksCubeRandomHLGNeuralHeuristic",
    "RubiksCubeRandomNeuralHeuristic",
    "SlidePuzzleConvNeuralHeuristic",
    "SlidePuzzleNeuralHeuristic",
    "SlidePuzzleRandomNeuralHeuristic",
    "SokobanNeuralHeuristic",
    "WorldModelNeuralHeuristic",
]

_LIGHTSOUT_MODULE = "heuristic.neuralheuristic.model.lightsout_neuralheuristic"
_PANCAKE_MODULE = "heuristic.neuralheuristic.model.pancake_neuralheuristic"
_RUBIKSCUBE_MODULE = "heuristic.neuralheuristic.model.rubikscube_neuralheuristic"
_SLIDEPUZZLE_MODULE = "heuristic.neuralheuristic.model.slidepuzzle_neuralheuristic"
_SOKOBAN_MODULE = "heuristic.neuralheuristic.model.sokoban_neuralheuristic"
_WORLD_MODEL_MODULE = "heuristic.neuralheuristic.model.world_model_neuralheuristic"

_EXPORTS = {
    "LightsOutConvNeuralHeuristic": _LIGHTSOUT_MODULE,
    "LightsOutNeuralHeuristic": _LIGHTSOUT_MODULE,
    "LightsOutRandomNeuralHeuristic": _LIGHTSOUT_MODULE,
    "PancakeNeuralHeuristic": _PANCAKE_MODULE,
    "PancakeRandomNeuralHeuristic": _PANCAKE_MODULE,
    "RubiksCubeHLGNeuralHeuristic": _RUBIKSCUBE_MODULE,
    "RubiksCubeNeuralHeuristic": _RUBIKSCUBE_MODULE,
    "RubiksCubeRandomHLGNeuralHeuristic": _RUBIKSCUBE_MODULE,
    "RubiksCubeRandomNeuralHeuristic": _RUBIKSCUBE_MODULE,
    "SlidePuzzleConvNeuralHeuristic": _SLIDEPUZZLE_MODULE,
    "SlidePuzzleNeuralHeuristic": _SLIDEPUZZLE_MODULE,
    "SlidePuzzleRandomNeuralHeuristic": _SLIDEPUZZLE_MODULE,
    "SokobanNeuralHeuristic": _SOKOBAN_MODULE,
    "WorldModelNeuralHeuristic": _WORLD_MODEL_MODULE,
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
