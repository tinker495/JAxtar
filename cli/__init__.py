from __future__ import annotations

__all__ = ["cli"]


def __getattr__(name: str):
    if name != "cli":
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    from .main import cli

    globals()[name] = cli
    return cli
