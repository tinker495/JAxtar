"""Unified JAxtar command line, parsed by tyro."""

import tyro

from .benchmark_commands import benchmark_app
from .commands import search_app
from .runtime import run_cli
from .train_commands import distance_train_app

app = tyro.extras.SubcommandApp()
app.command(search_app, name="test", help="Run individual search algorithms.")
app.command(benchmark_app, name="benchmark", help="Benchmark exact or generated workloads.")
app.command(distance_train_app, name="distance-train", help="Train neural distance estimators.")


def cli(args=None):
    return run_cli(
        app.cli,
        args,
        prog="jaxtar",
        description="JAxtar: JAX-based A* and Q* search for solving puzzles.",
    )


if __name__ == "__main__":
    cli()
