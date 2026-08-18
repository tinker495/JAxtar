from __future__ import annotations

from _lazy_imports import lazy_dir, load_lazy_export

__all__ = [
    "puzzle_bundles",
    "benchmark_bundles",
    "train_presets",
    "PuzzleOptions",
    "SearchOptions",
    "VisualizeOptions",
    "HeuristicOptions",
    "QFunctionOptions",
    "DistTrainOptions",
    "PuzzleBundle",
    "NeuralCallableConfig",
    "PuzzleConfig",
    "SEARCH_ALGORITHM_CATALOG",
    "SearchAlgorithmEntry",
    "resolve_algorithm_for_component",
]

_EXPORTS = {
    "SEARCH_ALGORITHM_CATALOG": ".algorithm_registry",
    "SearchAlgorithmEntry": ".algorithm_registry",
    "resolve_algorithm_for_component": ".algorithm_registry",
    "benchmark_bundles": ".benchmark_registry",
    "puzzle_bundles": ".puzzle_registry",
    "train_presets": ".train_presets",
    "DistTrainOptions": ".pydantic_models",
    "HeuristicOptions": ".pydantic_models",
    "NeuralCallableConfig": ".pydantic_models",
    "PuzzleBundle": ".pydantic_models",
    "PuzzleConfig": ".pydantic_models",
    "PuzzleOptions": ".pydantic_models",
    "QFunctionOptions": ".pydantic_models",
    "SearchOptions": ".pydantic_models",
    "VisualizeOptions": ".pydantic_models",
}


def __getattr__(name: str):
    return load_lazy_export(name, __name__, _EXPORTS, globals())


def __dir__() -> list[str]:
    return lazy_dir(globals(), __all__)
