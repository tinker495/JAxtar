from __future__ import annotations

import importlib
from typing import Any

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
