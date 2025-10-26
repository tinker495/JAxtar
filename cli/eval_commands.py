from datetime import datetime
from pathlib import Path
from typing import Callable, Optional, Union

import click
import matplotlib
from puxle import Puzzle
from rich.console import Console

from config.pydantic_models import EvalOptions, PuzzleOptions
from helpers.config_printer import print_config
from helpers.logger import BaseLogger
from heuristic.heuristic_base import Heuristic
from JAxtar.beamsearch.heuristic_beam import beam_builder
from JAxtar.beamsearch.q_beam import qbeam_builder
from JAxtar.stars.astar import astar_builder
from JAxtar.stars.qstar import qstar_builder
from qfunction.q_base import QFunction

from .comparison_generator import ComparisonGenerator
from .evaluation_runner import EvaluationRunner
from .options import (
    eval_options,
    eval_puzzle_options,
    heuristic_options,
    qfunction_options,
)

matplotlib.use("Agg")


@click.group(name="eval")
def evaluation():
    """Evaluation commands"""
    pass


def _run_evaluation_sweep(
    puzzle: Puzzle,
    puzzle_name: str,
    search_model: Union[Heuristic, QFunction],
    search_model_name: str,
    search_builder_fn: Callable,
    eval_options: EvalOptions,
    puzzle_opts: PuzzleOptions,
    output_dir: Optional[Path] = None,
    logger: Optional[BaseLogger] = None,
    step: int = 0,
    **kwargs,
):
    print_config(
        "Evaluation Configuration",
        {
            "puzzle_options": puzzle_opts.dict(),
            search_model_name: search_model.__class__.__name__,
            f"{search_model_name}_metadata": getattr(search_model, "metadata", {}),
            "eval_options": eval_options.dict(),
        },
    )
    runner = EvaluationRunner(
        puzzle=puzzle,
        puzzle_name=puzzle_name,
        search_model=search_model,
        search_model_name=search_model_name,
        search_builder_fn=search_builder_fn,
        eval_options=eval_options,
        puzzle_opts=puzzle_opts,
        output_dir=output_dir,
        logger=logger,
        step=step,
        **kwargs,
    )
    runner.run()


@evaluation.command(name="astar")
@eval_puzzle_options
@eval_options
@heuristic_options
def eval_astar(
    puzzle: Puzzle,
    puzzle_name: str,
    heuristic: Heuristic,
    eval_options: EvalOptions,
    puzzle_opts: PuzzleOptions,
    **kwargs,
):
    """Evaluate a heuristic-driven A* search with optional parameter sweeps."""
    _run_evaluation_sweep(
        puzzle=puzzle,
        puzzle_name=puzzle_name,
        search_model=heuristic,
        search_model_name="heuristic",
        search_builder_fn=astar_builder,
        eval_options=eval_options,
        puzzle_opts=puzzle_opts,
        **kwargs,
    )


@evaluation.command(name="beam")
@eval_puzzle_options
@eval_options(variant="beam")
@heuristic_options
def eval_beam(
    puzzle: Puzzle,
    puzzle_name: str,
    heuristic: Heuristic,
    eval_options: EvalOptions,
    puzzle_opts: PuzzleOptions,
    **kwargs,
):
    """Evaluate a heuristic-driven beam search with optional parameter sweeps."""
    _run_evaluation_sweep(
        puzzle=puzzle,
        puzzle_name=puzzle_name,
        search_model=heuristic,
        search_model_name="heuristic",
        search_builder_fn=beam_builder,
        eval_options=eval_options,
        puzzle_opts=puzzle_opts,
        node_metric_label="Beam Slots",
        **kwargs,
    )


@evaluation.command(name="qstar")
@eval_puzzle_options
@eval_options
@qfunction_options
def eval_qstar(
    puzzle: Puzzle,
    puzzle_name: str,
    qfunction: QFunction,
    eval_options: EvalOptions,
    puzzle_opts: PuzzleOptions,
    **kwargs,
):
    """Evaluate a Q*-style search with optional parameter sweeps."""
    _run_evaluation_sweep(
        puzzle=puzzle,
        puzzle_name=puzzle_name,
        search_model=qfunction,
        search_model_name="qfunction",
        search_builder_fn=qstar_builder,
        eval_options=eval_options,
        puzzle_opts=puzzle_opts,
        **kwargs,
    )


@evaluation.command(name="qbeam")
@eval_puzzle_options
@eval_options(variant="beam")
@qfunction_options
def eval_qbeam(
    puzzle: Puzzle,
    puzzle_name: str,
    qfunction: QFunction,
    eval_options: EvalOptions,
    puzzle_opts: PuzzleOptions,
    **kwargs,
):
    """Evaluate a Q-beam search with optional parameter sweeps."""
    _run_evaluation_sweep(
        puzzle=puzzle,
        puzzle_name=puzzle_name,
        search_model=qfunction,
        search_model_name="qfunction",
        search_builder_fn=qbeam_builder,
        eval_options=eval_options,
        puzzle_opts=puzzle_opts,
        node_metric_label="Beam Slots",
        **kwargs,
    )


@evaluation.command(name="compare")
@click.argument("run_dirs", nargs=-1, type=click.Path(exists=True, file_okay=False))
@click.option(
    "--scatter-max-points",
    type=int,
    default=2000,
    help="Maximum number of points to display on scatter plots.",
)
def eval_compare(run_dirs: list[str], scatter_max_points: int):
    """
    Compare multiple evaluation runs.

    Can accept individual run directories or parent directories containing multiple runs.
    """
    console = Console()
    if not run_dirs:
        console.print("[bold red]Error: Please provide at least one run directory.[/bold red]")
        return

    # Discover all individual run directories
    actual_run_dirs = []
    for run_dir_str in run_dirs:
        run_dir = Path(run_dir_str)
        # Check if the directory itself is a run directory
        if (run_dir / "results.csv").exists():
            actual_run_dirs.append(run_dir_str)
        else:
            # If not, treat it as a parent and search for sub-directories
            sub_dirs_found = [
                str(sub_dir)
                for sub_dir in run_dir.iterdir()
                if sub_dir.is_dir() and (sub_dir / "results.csv").exists()
            ]
            if sub_dirs_found:
                console.print(f"Found {len(sub_dirs_found)} sub-runs in [bold]{run_dir}[/bold]")
                actual_run_dirs.extend(sub_dirs_found)
            else:
                console.print(
                    f"[yellow]Warning: Directory {run_dir} is not a valid run and contains no sub-runs."
                    f"Skipping.[/yellow]"
                )

    if not actual_run_dirs:
        console.print("[bold red]Error: No valid run directories found to compare.[/bold red]")
        return

    # Determine output directory
    output_dir: Path
    if len(run_dirs) == 1 and Path(run_dirs[0]).is_dir():
        # If a single directory is provided (likely a sweep parent), save results there
        output_dir = Path(run_dirs[0])
    else:
        # For multiple dirs or other cases, create a new comparison directory
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        output_dir = Path("runs") / f"comparison_{timestamp}"

    output_dir.mkdir(parents=True, exist_ok=True)

    # Use set to avoid duplicates if user provides both parent and sub-run
    unique_run_dirs = sorted(list(set(actual_run_dirs)))

    comparison_generator = ComparisonGenerator(
        run_dirs=unique_run_dirs,
        output_dir=output_dir,
        scatter_max_points=scatter_max_points,
    )
    comparison_generator.generate_report()
    console.print(f"Comparison report saved in [bold]{output_dir}[/bold]")
