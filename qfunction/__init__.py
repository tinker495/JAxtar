from __future__ import annotations

import importlib
from typing import Any

__all__ = [
    "QFunction",
    "DotKnotQ",
    "EmptyQFunction",
    "LightsOutQ",
    "MazeQ",
    "LightsOutNeuralQ",
    "RubiksCubeNeuralQ",
    "SlidePuzzleNeuralQ",
    "PancakeQ",
    "PDDLQ",
    "RubiksCubeQ",
    "SlidePuzzleQ",
    "SokobanQ",
    "TSPQ",
]

_EXPORTS = {
    "QFunction": ("qfunction.q_base", "QFunction"),
    "DotKnotQ": ("qfunction.dotknot_q", "DotKnotQ"),
    "EmptyQFunction": ("qfunction.empty_q", "EmptyQFunction"),
    "LightsOutQ": ("qfunction.lightsout_q", "LightsOutQ"),
    "MazeQ": ("qfunction.maze_q", "MazeQ"),
    "LightsOutNeuralQ": ("qfunction.neuralq", "LightsOutNeuralQ"),
    "RubiksCubeNeuralQ": ("qfunction.neuralq", "RubiksCubeNeuralQ"),
    "SlidePuzzleNeuralQ": ("qfunction.neuralq", "SlidePuzzleNeuralQ"),
    "PancakeQ": ("qfunction.pancake_q", "PancakeQ"),
    "PDDLQ": ("qfunction.pddl_q", "PDDLQ"),
    "RubiksCubeQ": ("qfunction.rubikscube_q", "RubiksCubeQ"),
    "SlidePuzzleQ": ("qfunction.slidepuzzle_q", "SlidePuzzleQ"),
    "SokobanQ": ("qfunction.sokoban_q", "SokobanQ"),
    "TSPQ": ("qfunction.tsp_q", "TSPQ"),
}


def __getattr__(name: str) -> Any:
    try:
        module_name, attr_name = _EXPORTS[name]
    except KeyError as exc:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from exc
    value = getattr(importlib.import_module(module_name), attr_name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))
