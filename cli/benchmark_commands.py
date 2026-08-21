"""Benchmark commands generated from the Search Algorithm Catalog.

Each algorithm accepts either an exact ``--benchmark`` dataset or a generated
``--puzzle`` workload when no exact dataset exists.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import click
from rich.console import Console

from config.algorithm_registry import SEARCH_ALGORITHM_CATALOG, SearchAlgorithmEntry

from .comparison_generator import ComparisonGenerator
from .evaluation_runner import run_evaluation_sweep
from .options import (
    benchmark_options,
    eval_options,
    heuristic_options,
    qfunction_options,
)

benchmark = click.Group(
    name="benchmark",
    help="Benchmark search strategies with exact or generated workloads.",
)


def _build_benchmark_command(entry: SearchAlgorithmEntry) -> click.Command:
    component_dec = heuristic_options if entry.component_kind == "heuristic" else qfunction_options
    eval_dec = eval_options(variant="beam") if entry.is_beam else eval_options
    extra_sweep_kwargs = (
        {"node_metric_label": entry.node_metric_label} if entry.node_metric_label else {}
    )

    def inner(**kwargs):
        run_evaluation_sweep(
            puzzle=kwargs["puzzle"],
            puzzle_name=kwargs["puzzle_name"],
            search_model=kwargs[entry.component_kind],
            search_model_name=entry.component_kind,
            run_label=entry.python_id,
            search_builder_fn=entry.builder_fn,
            eval_options=kwargs["eval_options"],
            puzzle_opts=kwargs["puzzle_opts"],
            benchmark=kwargs.get("benchmark"),
            benchmark_name=kwargs.get("benchmark_name"),
            benchmark_bundle=kwargs.get("benchmark_bundle"),
            benchmark_cli_options=kwargs.get("benchmark_cli_options", {}),
            output_dir=kwargs["output_dir"],
            **extra_sweep_kwargs,
        )

    inner.__doc__ = entry.eval_description
    inner = click.option(
        "--output-dir",
        type=click.Path(path_type=Path),
        default=None,
        help="Directory to store run artifacts (defaults to runs/<timestamp>).",
    )(inner)
    inner = component_dec(inner)
    inner = eval_dec(inner)
    inner = benchmark_options(inner)
    return click.command(name=entry.cli_subcommand, help=entry.eval_description)(inner)


for _entry in SEARCH_ALGORITHM_CATALOG:
    benchmark.add_command(_build_benchmark_command(_entry))


@benchmark.command(name="compare")
@click.argument(
    "run_dirs",
    nargs=-1,
    required=True,
    type=click.Path(exists=True, file_okay=False),
)
@click.option(
    "--scatter-max-points",
    type=int,
    default=2000,
    help="Maximum number of points to display on scatter plots.",
)
def benchmark_compare(run_dirs: list[str], scatter_max_points: int):
    """Compare multiple benchmark runs."""
    console = Console()
    actual_run_dirs = []
    for run_dir_str in run_dirs:
        run_dir = Path(run_dir_str)
        if (run_dir / "results.csv").exists():
            actual_run_dirs.append(run_dir_str)
            continue
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

    if len(run_dirs) == 1 and Path(run_dirs[0]).is_dir():
        output_dir = Path(run_dirs[0])
    else:
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        output_dir = Path("runs") / f"comparison_{timestamp}"

    output_dir.mkdir(parents=True, exist_ok=True)
    comparison_generator = ComparisonGenerator(
        run_dirs=sorted(set(actual_run_dirs)),
        output_dir=output_dir,
        scatter_max_points=scatter_max_points,
    )
    comparison_generator.generate_report()
    console.print(f"Comparison report saved in [bold]{output_dir}[/bold]")


__all__ = ["benchmark"]
