from __future__ import annotations

import importlib
from typing import Any

__all__ = [
    "puzzle_bundles",
    "benchmark_bundles",
    "world_model_bundles",
    "train_presets",
    "PuzzleOptions",
    "SearchOptions",
    "VisualizeOptions",
    "HeuristicOptions",
    "QFunctionOptions",
    "DistTrainOptions",
    "WMDatasetOptions",
    "WMGetDSOptions",
    "WMGetModelOptions",
    "WMTrainOptions",
    "PuzzleBundle",
    "WorldModelBundle",
    "WorldModelPuzzleConfig",
    "NeuralCallableConfig",
    "PuzzleConfig",
    "SEARCH_ALGORITHM_CATALOG",
    "SearchAlgorithmEntry",
    "SearchAlgorithmResolution",
    "ComponentKind",
    "get_algorithm_entry",
    "resolve_algorithm_for_component",
]

_EXPORTS = {
    "SEARCH_ALGORITHM_CATALOG": "algorithm_registry",
    "ComponentKind": "algorithm_registry",
    "SearchAlgorithmEntry": "algorithm_registry",
    "SearchAlgorithmResolution": "algorithm_registry",
    "get_algorithm_entry": "algorithm_registry",
    "resolve_algorithm_for_component": "algorithm_registry",
    "benchmark_bundles": "benchmark_registry",
    "puzzle_bundles": "puzzle_registry",
    "train_presets": "train_presets",
    "world_model_bundles": "world_model_registry",
    "DistTrainOptions": "pydantic_models",
    "HeuristicOptions": "pydantic_models",
    "NeuralCallableConfig": "pydantic_models",
    "PuzzleBundle": "pydantic_models",
    "PuzzleConfig": "pydantic_models",
    "PuzzleOptions": "pydantic_models",
    "QFunctionOptions": "pydantic_models",
    "SearchOptions": "pydantic_models",
    "VisualizeOptions": "pydantic_models",
    "WMDatasetOptions": "pydantic_models",
    "WMGetDSOptions": "pydantic_models",
    "WMGetModelOptions": "pydantic_models",
    "WMTrainOptions": "pydantic_models",
    "WorldModelBundle": "pydantic_models",
    "WorldModelPuzzleConfig": "pydantic_models",
}


def __getattr__(name: str) -> Any:
    try:
        module_name = _EXPORTS[name]
    except KeyError as exc:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from exc
    value = getattr(importlib.import_module(f"config.{module_name}"), name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))
