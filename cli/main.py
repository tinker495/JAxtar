from __future__ import annotations

import click

from .benchmark_commands import benchmark
from .commands import SEARCH_COMMANDS
from .train_commands import distance_train


@click.group()
def cli():
    """JAxtar: A JAX-based A* and Q* search library for solving puzzles."""
    pass


search_test = click.Group(name="test", help="Run individual search algorithms.")


for _cmd in SEARCH_COMMANDS:
    search_test.add_command(_cmd)
cli.add_command(search_test)
cli.add_command(benchmark)
cli.add_command(distance_train)


if __name__ == "__main__":
    cli()
