import click

from .commands import astar, human_play, qstar
from .eval_commands import evaluation
from .train_commands import distance_train, world_model_train


@click.group()
def cli():
    """JAxtar: A JAX-based A* and Q* search library for solving puzzles."""
    pass


cli.add_command(astar)
cli.add_command(qstar)
cli.add_command(human_play)
cli.add_command(distance_train)
cli.add_command(world_model_train)
cli.add_command(evaluation)


if __name__ == "__main__":
    cli()
