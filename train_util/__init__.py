"""
Training utilities module for JAxtar.

Provides dataset generation, optimizer factories, and loss functions
for neural heuristic and Q-function training.
"""

from __future__ import annotations

import importlib
from types import ModuleType

__all__ = ["optimizer", "sampling", "losses"]

_MODULES = {name: f"{__name__}.{name}" for name in __all__}


def __getattr__(name: str) -> ModuleType:
    try:
        module_name = _MODULES[name]
    except KeyError as exc:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from exc
    module = importlib.import_module(module_name)
    globals()[name] = module
    return module


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))
