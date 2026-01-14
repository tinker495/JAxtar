import click

from .benchmark_commands import benchmark
from .commands import astar, astar_d, beam, id_astar, id_qstar, qbeam, qstar
from .eval_commands import evaluation
from .human_play import human_play
from .train_commands import distance_train, world_model_train


@click.group()
def cli():
    """JAxtar: A JAX-based A* and Q* search library for solving puzzles."""
    pass


cli.add_command(astar)
cli.add_command(astar_d)
cli.add_command(beam)
cli.add_command(id_astar)
cli.add_command(id_qstar)
cli.add_command(qbeam)
cli.add_command(qstar)
cli.add_command(human_play)
cli.add_command(distance_train)
cli.add_command(world_model_train)
cli.add_command(evaluation)
cli.add_command(benchmark)


if __name__ == "__main__":
    cli()
