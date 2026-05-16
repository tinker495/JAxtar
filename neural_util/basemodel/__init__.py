from __future__ import annotations

import importlib
from typing import Any

__all__ = ["DistanceHLGModel", "DistanceModel", "HLGResMLPModel", "ResMLPModel"]

_EXPORTS = {
    "DistanceHLGModel": "neural_util.basemodel.base",
    "DistanceModel": "neural_util.basemodel.base",
    "HLGResMLPModel": "neural_util.basemodel.hlgresmlp",
    "ResMLPModel": "neural_util.basemodel.resmlp",
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
