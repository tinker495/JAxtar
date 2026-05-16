from __future__ import annotations

import importlib
from typing import Any

__all__ = ["plot_search_tree_semantic"]


def __getattr__(name: str) -> Any:
    if name != "plot_search_tree_semantic":
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(importlib.import_module("helpers.plots.tree.plotting"), name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))
