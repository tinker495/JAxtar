from __future__ import annotations

from _lazy_imports import lazy_dir, load_lazy_export

__all__ = [
    "QFunction",
    "QFromHeuristic",
    "EmptyQFunction",
    "LightsOutNeuralQ",
    "RubiksCubeNeuralQ",
    "SlidePuzzleNeuralQ",
]

_EXPORTS = {
    "QFunction": ("qfunction.q_base", "QFunction"),
    "QFromHeuristic": ("qfunction.q_base", "QFromHeuristic"),
    "EmptyQFunction": ("qfunction.empty_q", "EmptyQFunction"),
    "LightsOutNeuralQ": ("qfunction.neuralq", "LightsOutNeuralQ"),
    "RubiksCubeNeuralQ": ("qfunction.neuralq", "RubiksCubeNeuralQ"),
    "SlidePuzzleNeuralQ": ("qfunction.neuralq", "SlidePuzzleNeuralQ"),
}


def __getattr__(name: str):
    return load_lazy_export(name, __name__, _EXPORTS, globals())


def __dir__() -> list[str]:
    return lazy_dir(globals(), __all__)
