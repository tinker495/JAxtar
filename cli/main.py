from __future__ import annotations

import click

from .commands import SEARCH_COMMANDS
from .human_play import human_play
from .lazy_group import LazyGroup
from .train_commands import distance_train, world_model_train

_LAZY_COMMANDS: dict[str, tuple[str, str, str]] = {
    "benchmark": (
        ".benchmark_commands",
        "benchmark",
        "Benchmark search strategies with registered configs.",
    ),
    "eval": (".eval_commands", "evaluation", "Evaluation commands."),
}


@click.group(cls=LazyGroup, lazy_commands=_LAZY_COMMANDS, import_package=__package__)
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
