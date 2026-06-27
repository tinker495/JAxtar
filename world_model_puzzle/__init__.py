from __future__ import annotations

from _lazy_imports import lazy_dir, load_lazy_export

__all__ = [
    "RubiksCubeWorldModel",
    "RubiksCubeWorldModel_reversed",
    "RubiksCubeWorldModel_test",
    "RubiksCubeWorldModelOptimized",
    "RubiksCubeWorldModelOptimized_reversed",
    "RubiksCubeWorldModelOptimized_test",
    "SokobanWorldModel",
    "SokobanWorldModelOptimized",
    "WorldModelPuzzleBase",
]

_RUBIKSCUBE_MODULE = "world_model_puzzle.model.rubikscube_world_model"
_SOKOBAN_MODULE = "world_model_puzzle.model.sokoban_world_model"
_BASE_MODULE = "world_model_puzzle.world_model_puzzle_base"

_EXPORTS = {
    "RubiksCubeWorldModel": _RUBIKSCUBE_MODULE,
    "RubiksCubeWorldModel_reversed": _RUBIKSCUBE_MODULE,
    "RubiksCubeWorldModel_test": _RUBIKSCUBE_MODULE,
    "RubiksCubeWorldModelOptimized": _RUBIKSCUBE_MODULE,
    "RubiksCubeWorldModelOptimized_reversed": _RUBIKSCUBE_MODULE,
    "RubiksCubeWorldModelOptimized_test": _RUBIKSCUBE_MODULE,
    "SokobanWorldModel": _SOKOBAN_MODULE,
    "SokobanWorldModelOptimized": _SOKOBAN_MODULE,
    "WorldModelPuzzleBase": _BASE_MODULE,
}


def __getattr__(name: str):
    return load_lazy_export(name, __name__, _EXPORTS, globals())


def __dir__() -> list[str]:
    return lazy_dir(globals(), __all__)
