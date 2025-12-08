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
from .options import benchmark_options, eval_options


@click.group(name="benchmark")
def benchmark() -> None:
    """Benchmark search strategies with registered configs."""


def _resolve_param_path(template: str, puzzle: Puzzle, override: Optional[str]) -> str:
    if override:
        return override
    if "{size}" in template:
        size = getattr(puzzle, "size", None)
        if size is None:
            raise click.UsageError(
                "Parameter path template requires 'size', but puzzle has no size attribute."
            )
        return template.format(size=size)
    return template


def _load_benchmark_heuristic(
    puzzle: Puzzle,
    benchmark_name: str,
    benchmark_bundle,
    param_path: Optional[str],
    model_type: str = "default",
) -> Heuristic:
    heuristic_configs = benchmark_bundle.heuristic_nn_configs
    if heuristic_configs is None:
        raise click.UsageError(
            f"Benchmark '{benchmark_name}' does not define a heuristic configuration."
        )

    heuristic_config = heuristic_configs.get(model_type)
    if heuristic_config is None:
        raise click.UsageError(
            f"Benchmark '{benchmark_name}' heuristic config type '{model_type}' not available."
        )

    path_template = heuristic_config.param_path
    if path_template is None:
        raise click.UsageError(
            f"Benchmark '{benchmark_name}' heuristic config '{model_type}' has no param path."
        )

    resolved_param_path = _resolve_param_path(path_template, puzzle, param_path)
    neural_config = {}

    return heuristic_config.callable(
        puzzle=puzzle,
        path=resolved_param_path,
        init_params=False,
        **neural_config,
    )


def _load_benchmark_qfunction(
    puzzle: Puzzle,
    benchmark_name: str,
    benchmark_bundle,
    param_path: Optional[str],
    model_type: str = "default",
) -> QFunction:
    q_configs = benchmark_bundle.q_function_nn_configs
    if q_configs is None:
        raise click.UsageError(
            f"Benchmark '{benchmark_name}' does not define a Q-function configuration."
        )

    q_config = q_configs.get(model_type)
    if q_config is None:
        raise click.UsageError(
            f"Benchmark '{benchmark_name}' Q-function config type '{model_type}' not available."
        )

    path_template = q_config.param_path
    if path_template is None:
        raise click.UsageError(
            f"Benchmark '{benchmark_name}' Q-function config '{model_type}' has no param path."
        )

    resolved_param_path = _resolve_param_path(path_template, puzzle, param_path)
    neural_config = {}

    return q_config.callable(
        puzzle=puzzle,
        path=resolved_param_path,
        init_params=False,
        **neural_config,
    )


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
@click.option(
    "--param-path",
    type=str,
    default=None,
    help="Optional override for the heuristic parameter file.",
)
@click.option(
    "--model-type",
    type=str,
    default="default",
    help="Type of the heuristic model (default: 'default').",
)
@click.option(
    "--output-dir",
    type=click.Path(path_type=Path),
    default=None,
    help="Directory to store run artifacts (defaults to runs/<timestamp>).",
)
def benchmark_astar(
    puzzle: Puzzle,
    eval_options: EvalOptions,
    benchmark,
    benchmark_name: str,
    benchmark_bundle,
    benchmark_cli_options,
    param_path: Optional[str],
    model_type: str,
    output_dir: Optional[Path],
    **kwargs,
):
    solver = _load_benchmark_heuristic(
        puzzle, benchmark_name, benchmark_bundle, param_path, model_type
    )
    puzzle_opts = PuzzleOptions(puzzle=benchmark_name)
    _run_benchmark(
        puzzle=puzzle,
        puzzle_name=benchmark_name,
        search_model=solver,
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
@click.option(
    "--param-path",
    type=str,
    default=None,
    help="Optional override for the heuristic parameter file.",
)
@click.option(
    "--model-type",
    type=str,
    default="default",
    help="Type of the heuristic model (default: 'default').",
)
@click.option(
    "--output-dir",
    type=click.Path(path_type=Path),
    default=None,
    help="Directory to store run artifacts (defaults to runs/<timestamp>).",
)
def benchmark_astar_d(
    puzzle: Puzzle,
    eval_options: EvalOptions,
    benchmark,
    benchmark_name: str,
    benchmark_bundle,
    benchmark_cli_options,
    param_path: Optional[str],
    model_type: str,
    output_dir: Optional[Path],
    **kwargs,
):
    solver = _load_benchmark_heuristic(
        puzzle, benchmark_name, benchmark_bundle, param_path, model_type
    )
    puzzle_opts = PuzzleOptions(puzzle=benchmark_name)
    _run_benchmark(
        puzzle=puzzle,
        puzzle_name=benchmark_name,
        search_model=solver,
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
@click.option(
    "--param-path",
    type=str,
    default=None,
    help="Optional override for the Q-function parameter file.",
)
@click.option(
    "--model-type",
    type=str,
    default="default",
    help="Type of the Q-function model (default: 'default').",
)
@click.option(
    "--output-dir",
    type=click.Path(path_type=Path),
    default=None,
    help="Directory to store run artifacts (defaults to runs/<timestamp>).",
)
def benchmark_qstar(
    puzzle: Puzzle,
    eval_options: EvalOptions,
    benchmark,
    benchmark_name: str,
    benchmark_bundle,
    benchmark_cli_options,
    param_path: Optional[str],
    model_type: str,
    output_dir: Optional[Path],
    **kwargs,
):
    solver = _load_benchmark_qfunction(
        puzzle, benchmark_name, benchmark_bundle, param_path, model_type
    )
    puzzle_opts = PuzzleOptions(puzzle=benchmark_name)
    _run_benchmark(
        puzzle=puzzle,
        puzzle_name=benchmark_name,
        search_model=solver,
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
@click.option(
    "--param-path",
    type=str,
    default=None,
    help="Optional override for the heuristic parameter file.",
)
@click.option(
    "--model-type",
    type=str,
    default="default",
    help="Type of the heuristic model (default: 'default').",
)
@click.option(
    "--output-dir",
    type=click.Path(path_type=Path),
    default=None,
    help="Directory to store run artifacts (defaults to runs/<timestamp>).",
)
def benchmark_beam(
    puzzle: Puzzle,
    eval_options: EvalOptions,
    benchmark,
    benchmark_name: str,
    benchmark_bundle,
    benchmark_cli_options,
    param_path: Optional[str],
    model_type: str,
    output_dir: Optional[Path],
    **kwargs,
):
    solver = _load_benchmark_heuristic(
        puzzle, benchmark_name, benchmark_bundle, param_path, model_type
    )
    puzzle_opts = PuzzleOptions(puzzle=benchmark_name)
    _run_benchmark(
        puzzle=puzzle,
        puzzle_name=benchmark_name,
        search_model=solver,
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
@click.option(
    "--param-path",
    type=str,
    default=None,
    help="Optional override for the Q-function parameter file.",
)
@click.option(
    "--model-type",
    type=str,
    default="default",
    help="Type of the Q-function model (default: 'default').",
)
@click.option(
    "--output-dir",
    type=click.Path(path_type=Path),
    default=None,
    help="Directory to store run artifacts (defaults to runs/<timestamp>).",
)
def benchmark_qbeam(
    puzzle: Puzzle,
    eval_options: EvalOptions,
    benchmark,
    benchmark_name: str,
    benchmark_bundle,
    benchmark_cli_options,
    param_path: Optional[str],
    model_type: str,
    output_dir: Optional[Path],
    **kwargs,
):
    solver = _load_benchmark_qfunction(
        puzzle, benchmark_name, benchmark_bundle, param_path, model_type
    )
    puzzle_opts = PuzzleOptions(puzzle=benchmark_name)
    _run_benchmark(
        puzzle=puzzle,
        puzzle_name=benchmark_name,
        search_model=solver,
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
