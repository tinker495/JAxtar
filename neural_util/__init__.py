"""Neural utilities module for JAxtar."""

from __future__ import annotations

import importlib
from typing import Any

__all__ = ["modules", "aqt_utils", "param_manager", "nn_metadata", "norm"]


def __getattr__(name: str) -> Any:
    if name not in __all__:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = importlib.import_module(f"neural_util.{name}")
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))
