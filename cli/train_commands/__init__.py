from __future__ import annotations

import importlib
from typing import Any

import click

from _lazy_imports import lazy_dir, load_lazy_export

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


class LazyGroup(click.Group):
    def __init__(self, *args: Any, command_exports: dict[str, tuple[str, str]], **kwargs: Any):
        super().__init__(*args, **kwargs)
        self._command_exports = command_exports

    def list_commands(self, ctx: click.Context) -> list[str]:
        names = set(super().list_commands(ctx))
        names.update(self._command_exports)
        return sorted(names)

    def get_command(self, ctx: click.Context, cmd_name: str) -> click.Command | None:
        command = super().get_command(ctx, cmd_name)
        if command is not None:
            return command
        spec = self._command_exports.get(cmd_name)
        if spec is None:
            return None
        module_name, attr_name = spec
        command = getattr(importlib.import_module(module_name), attr_name)
        self.add_command(command, name=cmd_name)
        return command


distance_train = LazyGroup(
    name="distance-train",
    help="Train neural heuristic and Q-function distance estimators.",
    command_exports={
        "heuristic": _COMMAND_EXPORTS["heuristic_train_command"],
        "qfunction": _COMMAND_EXPORTS["qfunction_train_command"],
    },
)
world_model_train = LazyGroup(
    name="world-model-train",
    help="Create datasets and train world models.",
    command_exports={
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
