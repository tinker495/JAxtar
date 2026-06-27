from __future__ import annotations

from _lazy_imports import lazy_dir, load_lazy_export

__all__ = ["DistanceHLGModel", "DistanceModel", "HLGResMLPModel", "ResMLPModel"]

_EXPORTS = {
    "DistanceHLGModel": "neural_util.basemodel.base",
    "DistanceModel": "neural_util.basemodel.base",
    "HLGResMLPModel": "neural_util.basemodel.hlgresmlp",
    "ResMLPModel": "neural_util.basemodel.resmlp",
}


def __getattr__(name: str):
    return load_lazy_export(name, __name__, _EXPORTS, globals())


def __dir__() -> list[str]:
    return lazy_dir(globals(), __all__)
