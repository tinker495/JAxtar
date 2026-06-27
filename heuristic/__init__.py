from __future__ import annotations

from _lazy_imports import lazy_dir, load_lazy_export

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


def __getattr__(name: str):
    return load_lazy_export(name, __name__, _EXPORTS, globals())


def __dir__() -> list[str]:
    return lazy_dir(globals(), __all__)
