from __future__ import annotations

import importlib
from typing import Any

__all__ = [
    "Heuristic",
    "DotKnotHeuristic",
    "EmptyHeuristic",
    "LightsOutHeuristic",
    "MazeHeuristic",
    "LightsOutNeuralHeuristic",
    "RubiksCubeNeuralHeuristic",
    "SlidePuzzleNeuralHeuristic",
    "PancakeHeuristic",
    "PDDLHeuristic",
    "RubiksCubeHeuristic",
    "SlidePuzzleHeuristic",
    "SokobanHeuristic",
    "TSPHeuristic",
]

_EXPORTS = {
    "Heuristic": ("heuristic.heuristic_base", "Heuristic"),
    "DotKnotHeuristic": ("heuristic.dotknot_heuristic", "DotKnotHeuristic"),
    "EmptyHeuristic": ("heuristic.empty_heuristic", "EmptyHeuristic"),
    "LightsOutHeuristic": ("heuristic.lightsout_heuristic", "LightsOutHeuristic"),
    "MazeHeuristic": ("heuristic.maze_heuristic", "MazeHeuristic"),
    "LightsOutNeuralHeuristic": (
        "heuristic.neuralheuristic",
        "LightsOutNeuralHeuristic",
    ),
    "RubiksCubeNeuralHeuristic": (
        "heuristic.neuralheuristic",
        "RubiksCubeNeuralHeuristic",
    ),
    "SlidePuzzleNeuralHeuristic": (
        "heuristic.neuralheuristic",
        "SlidePuzzleNeuralHeuristic",
    ),
    "PancakeHeuristic": ("heuristic.pancake_heuristic", "PancakeHeuristic"),
    "PDDLHeuristic": ("heuristic.pddl_heuristic", "PDDLHeuristic"),
    "RubiksCubeHeuristic": ("heuristic.rubikscube_heuristic", "RubiksCubeHeuristic"),
    "SlidePuzzleHeuristic": ("heuristic.slidepuzzle_heuristic", "SlidePuzzleHeuristic"),
    "SokobanHeuristic": ("heuristic.sokoban_heuristic", "SokobanHeuristic"),
    "TSPHeuristic": ("heuristic.tsp_heuristic", "TSPHeuristic"),
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
