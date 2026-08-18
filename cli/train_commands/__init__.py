from __future__ import annotations

from _lazy_imports import lazy_dir, load_lazy_export

from ..lazy_group import LazyGroup

__all__ = [
    "distance_train",
    "heuristic_train_command",
    "qfunction_train_command",
]

_COMMAND_EXPORTS = {
    "heuristic_train_command": (
        "cli.train_commands.dist_train_command",
        "heuristic_train_command",
    ),
    "qfunction_train_command": (
        "cli.train_commands.dist_train_command",
        "qfunction_train_command",
    ),
}


distance_train = LazyGroup(
    name="distance-train",
    help="Train neural heuristic and Q-function distance estimators.",
    lazy_commands={
        "heuristic": _COMMAND_EXPORTS["heuristic_train_command"],
        "qfunction": _COMMAND_EXPORTS["qfunction_train_command"],
    },
)


def __getattr__(name: str):
    return load_lazy_export(name, __name__, _COMMAND_EXPORTS, globals())


def __dir__() -> list[str]:
    return lazy_dir(globals(), __all__)
