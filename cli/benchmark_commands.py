"""Tyro benchmark commands for exact datasets and generated workloads."""

from dataclasses import asdict
from datetime import datetime
from pathlib import Path

import tyro
from rich.console import Console

from config.algorithm_registry import SEARCH_ALGORITHM_CATALOG, SearchAlgorithmEntry

from .comparison_generator import ComparisonGenerator
from .evaluation_runner import run_evaluation_sweep
from .options import (
    HeuristicBenchmarkArgs,
    QFunctionBenchmarkArgs,
    resolve_benchmark,
    resolve_eval,
    resolve_heuristic,
    resolve_qfunction,
)
from .runtime import run_cli

benchmark_app = tyro.extras.SubcommandApp()


def _build_benchmark_command(entry: SearchAlgorithmEntry):
    schema = (
        HeuristicBenchmarkArgs if entry.component_kind == "heuristic" else QFunctionBenchmarkArgs
    )
    extra_sweep_kwargs = (
        {"node_metric_label": entry.node_metric_label} if entry.node_metric_label else {}
    )

    def run(options):
        kwargs = resolve_benchmark(asdict(options))
        kwargs = resolve_eval(kwargs, variant="beam" if entry.is_beam else "default")
        resolver = resolve_heuristic if entry.component_kind == "heuristic" else resolve_qfunction
        kwargs = resolver(kwargs)
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

    run.__annotations__ = {"options": schema}
    run.__name__ = entry.python_id
    run.__doc__ = entry.eval_description
    benchmark_app.command(run, name=entry.cli_subcommand)


for _entry in SEARCH_ALGORITHM_CATALOG:
    _build_benchmark_command(_entry)


@benchmark_app.command(name="compare")
def benchmark_compare(
    run_dirs: tyro.conf.Positional[tuple[Path, ...]], scatter_max_points: int = 2000
):
    """Compare multiple benchmark runs."""
    if not run_dirs:
        raise ValueError("At least one run directory is required.")
    for directory in run_dirs:
        if not directory.is_dir():
            raise ValueError(f"Run directory does not exist: {directory}")
    console = Console()
    actual_run_dirs = []
    for run_dir_str in run_dirs:
        run_dir = Path(run_dir_str)
        if (run_dir / "results.csv").exists():
            actual_run_dirs.append(str(run_dir_str))
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


def benchmark(args=None):
    return run_cli(
        benchmark_app.cli,
        args,
        prog="benchmark",
        description="Benchmark search strategies with exact or generated workloads.",
    )


__all__ = ["benchmark", "benchmark_app"]
