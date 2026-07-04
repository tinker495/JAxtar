from __future__ import annotations

import importlib
from typing import Any, Mapping

import click

LazyCommandSpec = tuple[str, str] | tuple[str, str, str]


class LazyGroup(click.Group):
    def __init__(
        self,
        *args: Any,
        lazy_commands: Mapping[str, LazyCommandSpec],
        import_package: str | None = None,
        **kwargs: Any,
    ):
        super().__init__(*args, **kwargs)
        self._lazy_commands = dict(lazy_commands)
        self._import_package = import_package

    def list_commands(self, ctx: click.Context) -> list[str]:
        return sorted(set(super().list_commands(ctx)) | set(self._lazy_commands))

    def get_command(self, ctx: click.Context, cmd_name: str) -> click.Command | None:
        command = super().get_command(ctx, cmd_name)
        if command is not None:
            return command
        spec = self._lazy_commands.get(cmd_name)
        if spec is None:
            return None
        module_name, attr_name = spec[:2]
        command = getattr(importlib.import_module(module_name, self._import_package), attr_name)
        self.add_command(command, name=cmd_name)
        return command

    def format_commands(self, ctx: click.Context, formatter: click.HelpFormatter) -> None:
        rows = []
        for name in self.list_commands(ctx):
            if name in self._lazy_commands and name not in self.commands:
                spec = self._lazy_commands[name]
                if len(spec) == 3:
                    rows.append((name, spec[2]))
                    continue
            command = self.get_command(ctx, name)
            if command is None or command.hidden:
                continue
            rows.append((name, command.get_short_help_str()))
        if rows:
            with formatter.section("Commands"):
                formatter.write_dl(rows)
