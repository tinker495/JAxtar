"""Evaluation Click commands generated from the Search Algorithm Catalog.

See CONTEXT.md "Search Algorithm Catalog". The 10 algorithm-specific eval
subcommands (`eval astar`, `eval astar-d`, ...) are built by iterating
`SEARCH_ALGORITHM_CATALOG` and attaching to the `evaluation` Click group.
The `eval compare` command is not algorithm-dispatched and stays hand-written.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import click
import matplotlib
from rich.console import Console

from config.algorithm_registry import SEARCH_ALGORITHM_CATALOG, SearchAlgorithmEntry

from .comparison_generator import ComparisonGenerator
from .evaluation_runner import run_evaluation_sweep
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


def _build_eval_command(entry: SearchAlgorithmEntry) -> click.Command:
    component_dec = heuristic_options if entry.component_kind == "heuristic" else qfunction_options
    eval_dec = eval_options(variant="beam") if entry.is_beam else eval_options
    extra_sweep_kwargs = (
        {"node_metric_label": entry.node_metric_label} if entry.node_metric_label else {}
    )

    def inner(**kwargs):
        component = kwargs[entry.component_kind]
        run_evaluation_sweep(
            puzzle=kwargs["puzzle"],
            puzzle_name=kwargs["puzzle_name"],
            search_model=component,
            search_model_name=entry.component_kind,
            run_label=entry.run_label,
            search_builder_fn=entry.builder_fn,
            eval_options=kwargs["eval_options"],
            puzzle_opts=kwargs["puzzle_opts"],
            **extra_sweep_kwargs,
        )

    inner.__doc__ = entry.eval_description
    inner = component_dec(inner)
    inner = eval_dec(inner)
    inner = eval_puzzle_options(inner)
    return click.command(name=entry.cli_subcommand, help=entry.eval_description)(inner)


for _entry in SEARCH_ALGORITHM_CATALOG:
    evaluation.add_command(_build_eval_command(_entry))


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

    actual_run_dirs = []
    for run_dir_str in run_dirs:
        run_dir = Path(run_dir_str)
        if (run_dir / "results.csv").exists():
            actual_run_dirs.append(run_dir_str)
        else:
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

    output_dir: Path
    if len(run_dirs) == 1 and Path(run_dirs[0]).is_dir():
        output_dir = Path(run_dirs[0])
    else:
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        output_dir = Path("runs") / f"comparison_{timestamp}"

    output_dir.mkdir(parents=True, exist_ok=True)

    unique_run_dirs = sorted(list(set(actual_run_dirs)))

    comparison_generator = ComparisonGenerator(
        run_dirs=unique_run_dirs,
        output_dir=output_dir,
        scatter_max_points=scatter_max_points,
    )
    comparison_generator.generate_report()
    console.print(f"Comparison report saved in [bold]{output_dir}[/bold]")
