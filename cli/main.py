from __future__ import annotations

import importlib
from typing import Any

import click

from .commands import SEARCH_COMMANDS
from .human_play import human_play
from .train_commands import distance_train, world_model_train

_LAZY_COMMANDS: dict[str, tuple[str, str, str]] = {
    "benchmark": (
        ".benchmark_commands",
        "benchmark",
        "Benchmark search strategies with registered configs.",
    ),
    "eval": (".eval_commands", "evaluation", "Evaluation commands."),
}


class LazyRootGroup(click.Group):
    def __init__(self, *args: Any, lazy_commands: dict[str, tuple[str, str, str]], **kwargs: Any):
        super().__init__(*args, **kwargs)
        self._lazy_commands = lazy_commands

    def list_commands(self, ctx: click.Context) -> list[str]:
        return sorted(set(super().list_commands(ctx)) | set(self._lazy_commands))

    def get_command(self, ctx: click.Context, cmd_name: str) -> click.Command | None:
        command = super().get_command(ctx, cmd_name)
        if command is not None:
            return command
        spec = self._lazy_commands.get(cmd_name)
        if spec is None:
            return None
        module_name, attr_name, _help = spec
        command = getattr(importlib.import_module(module_name, __package__), attr_name)
        self.add_command(command, name=cmd_name)
        return command

    def format_commands(self, ctx: click.Context, formatter: click.HelpFormatter) -> None:
        rows = []
        for name in self.list_commands(ctx):
            if name in self._lazy_commands and name not in self.commands:
                rows.append((name, self._lazy_commands[name][2]))
                continue
            command = super().get_command(ctx, name)
            if command is None or command.hidden:
                continue
            rows.append((name, command.get_short_help_str()))
        if rows:
            with formatter.section("Commands"):
                formatter.write_dl(rows)


@click.group(cls=LazyRootGroup, lazy_commands=_LAZY_COMMANDS)
def cli():
    """JAxtar: A JAX-based A* and Q* search library for solving puzzles."""
    pass


for _cmd in SEARCH_COMMANDS:
    cli.add_command(_cmd)
cli.add_command(human_play)
cli.add_command(distance_train)
cli.add_command(world_model_train)


if __name__ == "__main__":
    cli()
