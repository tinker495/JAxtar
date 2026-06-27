from __future__ import annotations

from _lazy_imports import lazy_dir, load_lazy_export

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


def __getattr__(name: str):
    return load_lazy_export(name, __name__, _EXPORTS, globals())


def __dir__() -> list[str]:
    return lazy_dir(globals(), __all__)
