from __future__ import annotations

import importlib
from typing import Any

__all__ = ["cli"]


def __getattr__(name: str) -> Any:
    if name != "cli":
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(importlib.import_module("cli.main"), name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))
