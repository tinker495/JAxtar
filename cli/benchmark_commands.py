from __future__ import annotations

from pathlib import Path
from typing import Optional

import click
from puxle import Puzzle

from config.pydantic_models import EvalOptions, PuzzleOptions
from heuristic.heuristic_base import Heuristic
from JAxtar.beamsearch.heuristic_beam import beam_builder
from JAxtar.beamsearch.q_beam import qbeam_builder
from JAxtar.stars.astar import astar_builder
from JAxtar.stars.astar_d import astar_d_builder
from JAxtar.stars.qstar import qstar_builder
from qfunction.q_base import QFunction

from .evaluation_runner import run_evaluation_sweep
from .options import (
    benchmark_options,
    eval_options,
    heuristic_options,
    qfunction_options,
)


@click.group(name="benchmark")
def benchmark() -> None:
    """Benchmark search strategies with registered configs."""


def _run_benchmark(
    *,
    puzzle: Puzzle,
    puzzle_name: str,
    search_model,
    search_model_name: str,
    search_builder_fn,
    eval_options: EvalOptions,
    puzzle_opts: PuzzleOptions,
    benchmark,
    benchmark_name: str,
    benchmark_bundle,
    benchmark_cli_options,
    run_label: Optional[str] = None,
    output_dir: Optional[Path] = None,
    logger=None,
    step: int = 0,
) -> None:
    run_evaluation_sweep(
        puzzle=puzzle,
        puzzle_name=puzzle_name,
        search_model=search_model,
        search_model_name=search_model_name,
        run_label=run_label,
        search_builder_fn=search_builder_fn,
        eval_options=eval_options,
        puzzle_opts=puzzle_opts,
        output_dir=output_dir,
        logger=logger,
        step=step,
        benchmark=benchmark,
        benchmark_name=benchmark_name,
        benchmark_bundle=benchmark_bundle,
        benchmark_cli_options=benchmark_cli_options,
    )


@benchmark.command(name="astar")
@benchmark_options
@eval_options
@heuristic_options
@click.option(
    "--output-dir",
    type=click.Path(path_type=Path),
    default=None,
    help="Directory to store run artifacts (defaults to runs/<timestamp>).",
)
def benchmark_astar(
    puzzle: Puzzle,
    eval_options: EvalOptions,
    heuristic: Heuristic,
    benchmark,
    benchmark_name: str,
    benchmark_bundle,
    benchmark_cli_options,
    output_dir: Optional[Path],
    **kwargs,
):
    puzzle_opts = PuzzleOptions(puzzle=benchmark_name)
    _run_benchmark(
        puzzle=puzzle,
        puzzle_name=benchmark_name,
        search_model=heuristic,
        search_model_name="heuristic",
        run_label="astar",
        search_builder_fn=astar_builder,
        eval_options=eval_options,
        puzzle_opts=puzzle_opts,
        benchmark=benchmark,
        benchmark_name=benchmark_name,
        benchmark_bundle=benchmark_bundle,
        benchmark_cli_options=benchmark_cli_options,
        output_dir=output_dir,
    )


@benchmark.command(name="astar_d")
@benchmark_options
@eval_options
@heuristic_options
@click.option(
    "--output-dir",
    type=click.Path(path_type=Path),
    default=None,
    help="Directory to store run artifacts (defaults to runs/<timestamp>).",
)
def benchmark_astar_d(
    puzzle: Puzzle,
    eval_options: EvalOptions,
    heuristic: Heuristic,
    benchmark,
    benchmark_name: str,
    benchmark_bundle,
    benchmark_cli_options,
    output_dir: Optional[Path],
    **kwargs,
):
    puzzle_opts = PuzzleOptions(puzzle=benchmark_name)
    _run_benchmark(
        puzzle=puzzle,
        puzzle_name=benchmark_name,
        search_model=heuristic,
        search_model_name="heuristic",
        run_label="astar_d",
        search_builder_fn=astar_d_builder,
        eval_options=eval_options,
        puzzle_opts=puzzle_opts,
        benchmark=benchmark,
        benchmark_name=benchmark_name,
        benchmark_bundle=benchmark_bundle,
        benchmark_cli_options=benchmark_cli_options,
        output_dir=output_dir,
    )


@benchmark.command(name="qstar")
@benchmark_options
@eval_options
@qfunction_options
@click.option(
    "--output-dir",
    type=click.Path(path_type=Path),
    default=None,
    help="Directory to store run artifacts (defaults to runs/<timestamp>).",
)
def benchmark_qstar(
    puzzle: Puzzle,
    eval_options: EvalOptions,
    qfunction: QFunction,
    benchmark,
    benchmark_name: str,
    benchmark_bundle,
    benchmark_cli_options,
    output_dir: Optional[Path],
    **kwargs,
):
    puzzle_opts = PuzzleOptions(puzzle=benchmark_name)
    _run_benchmark(
        puzzle=puzzle,
        puzzle_name=benchmark_name,
        search_model=qfunction,
        search_model_name="qfunction",
        run_label="qstar",
        search_builder_fn=qstar_builder,
        eval_options=eval_options,
        puzzle_opts=puzzle_opts,
        benchmark=benchmark,
        benchmark_name=benchmark_name,
        benchmark_bundle=benchmark_bundle,
        benchmark_cli_options=benchmark_cli_options,
        output_dir=output_dir,
    )


@benchmark.command(name="beam")
@benchmark_options
@eval_options(variant="beam")
@heuristic_options
@click.option(
    "--output-dir",
    type=click.Path(path_type=Path),
    default=None,
    help="Directory to store run artifacts (defaults to runs/<timestamp>).",
)
def benchmark_beam(
    puzzle: Puzzle,
    eval_options: EvalOptions,
    heuristic: Heuristic,
    benchmark,
    benchmark_name: str,
    benchmark_bundle,
    benchmark_cli_options,
    output_dir: Optional[Path],
    **kwargs,
):
    puzzle_opts = PuzzleOptions(puzzle=benchmark_name)
    _run_benchmark(
        puzzle=puzzle,
        puzzle_name=benchmark_name,
        search_model=heuristic,
        search_model_name="heuristic",
        run_label="beam",
        search_builder_fn=beam_builder,
        eval_options=eval_options,
        puzzle_opts=puzzle_opts,
        benchmark=benchmark,
        benchmark_name=benchmark_name,
        benchmark_bundle=benchmark_bundle,
        benchmark_cli_options=benchmark_cli_options,
        output_dir=output_dir,
    )


@benchmark.command(name="qbeam")
@benchmark_options
@eval_options(variant="beam")
@qfunction_options
@click.option(
    "--output-dir",
    type=click.Path(path_type=Path),
    default=None,
    help="Directory to store run artifacts (defaults to runs/<timestamp>).",
)
def benchmark_qbeam(
    puzzle: Puzzle,
    eval_options: EvalOptions,
    qfunction: QFunction,
    benchmark,
    benchmark_name: str,
    benchmark_bundle,
    benchmark_cli_options,
    output_dir: Optional[Path],
    **kwargs,
):
    puzzle_opts = PuzzleOptions(puzzle=benchmark_name)
    _run_benchmark(
        puzzle=puzzle,
        puzzle_name=benchmark_name,
        search_model=qfunction,
        search_model_name="qfunction",
        run_label="qbeam",
        search_builder_fn=qbeam_builder,
        eval_options=eval_options,
        puzzle_opts=puzzle_opts,
        benchmark=benchmark,
        benchmark_name=benchmark_name,
        benchmark_bundle=benchmark_bundle,
        benchmark_cli_options=benchmark_cli_options,
        output_dir=output_dir,
    )


__all__ = [
    "benchmark",
    "benchmark_astar",
    "benchmark_astar_d",
    "benchmark_qstar",
    "benchmark_beam",
    "benchmark_qbeam",
]
