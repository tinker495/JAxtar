import click

from .benchmark_commands import benchmark
from .commands import SEARCH_COMMANDS
from .eval_commands import evaluation
from .human_play import human_play
from .train_commands import distance_train, world_model_train


@click.group()
def cli():
    """JAxtar: A JAX-based A* and Q* search library for solving puzzles."""
    pass


for _cmd in SEARCH_COMMANDS:
    cli.add_command(_cmd)
cli.add_command(human_play)
cli.add_command(distance_train)
cli.add_command(world_model_train)
cli.add_command(evaluation)
cli.add_command(benchmark)


if __name__ == "__main__":
    cli()
