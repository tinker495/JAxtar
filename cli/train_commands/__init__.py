from __future__ import annotations

from _lazy_imports import lazy_dir, load_lazy_export

from ..lazy_group import LazyGroup

__all__ = [
    "distance_train",
    "world_model_train",
    "heuristic_train_command",
    "qfunction_train_command",
    "make_puzzle_eval_trajectory",
    "make_puzzle_sample_data",
    "make_puzzle_transition_dataset",
    "train",
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
    "make_puzzle_eval_trajectory": (
        "cli.train_commands.world_model_ds_command",
        "make_puzzle_eval_trajectory",
    ),
    "make_puzzle_sample_data": (
        "cli.train_commands.world_model_ds_command",
        "make_puzzle_sample_data",
    ),
    "make_puzzle_transition_dataset": (
        "cli.train_commands.world_model_ds_command",
        "make_puzzle_transition_dataset",
    ),
    "train": ("cli.train_commands.world_model_train_command", "train"),
}


distance_train = LazyGroup(
    name="distance-train",
    help="Train neural heuristic and Q-function distance estimators.",
    lazy_commands={
        "heuristic": _COMMAND_EXPORTS["heuristic_train_command"],
        "qfunction": _COMMAND_EXPORTS["qfunction_train_command"],
    },
)
world_model_train = LazyGroup(
    name="world-model-train",
    help="Create datasets and train world models.",
    lazy_commands={
        "make_transition_dataset": _COMMAND_EXPORTS["make_puzzle_transition_dataset"],
        "make_sample_data": _COMMAND_EXPORTS["make_puzzle_sample_data"],
        "make_eval_trajectory": _COMMAND_EXPORTS["make_puzzle_eval_trajectory"],
        "train": _COMMAND_EXPORTS["train"],
    },
)


def __getattr__(name: str):
    return load_lazy_export(name, __name__, _COMMAND_EXPORTS, globals())


def __dir__() -> list[str]:
    return lazy_dir(globals(), __all__)
