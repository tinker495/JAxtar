from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

import click
from puxle import Puzzle

from config.pydantic_models import EvalOptions, PuzzleOptions
from helpers.param_stats import attach_runtime_metadata
from JAxtar.beamsearch.heuristic_beam import beam_builder
from JAxtar.beamsearch.q_beam import qbeam_builder
from JAxtar.bi_stars.bi_astar import bi_astar_builder
from JAxtar.bi_stars.bi_astar_d import bi_astar_d_builder
from JAxtar.bi_stars.bi_qstar import bi_qstar_builder
from JAxtar.id_stars.id_astar import id_astar_builder
from JAxtar.id_stars.id_qstar import id_qstar_builder
from JAxtar.stars.astar import astar_builder
from JAxtar.stars.astar_d import astar_d_builder
from JAxtar.stars.qstar import qstar_builder

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


def _benchmark_model_options(func):
    options = [
        click.option(
            "--param-path",
            type=str,
            default=None,
            help="Optional override for the parameter file.",
        ),
        click.option(
            "--model-type",
            type=str,
            default="default",
            help="Type of the model (default: 'default').",
        ),
        click.option(
            "-q",
            "--use-quantize",
            is_flag=True,
            default=False,
            help="Use quantization (defaults to int8).",
        ),
        click.option(
            "--quant-type",
            type=click.Choice(["int8", "int4", "int4_w8a", "int8_w_only"]),
            default="int8",
            help="Specific AQT quantization configuration to use.",
        ),
        click.option(
            "--output-dir",
            type=click.Path(path_type=Path),
            default=None,
            help="Directory to store run artifacts (defaults to runs/<timestamp>).",
        ),
    ]
    for opt in reversed(options):
        func = opt(func)
    return func


def _load_benchmark_solver(
    puzzle: Puzzle,
    benchmark_name: str,
    benchmark_bundle,
    param_path: Optional[str],
    model_type: str,
    aqt_cfg: Optional[str],
    search_model_name: str,
) -> Any:
    if search_model_name == "heuristic":
        configs = benchmark_bundle.heuristic_nn_configs
        config_desc = "heuristic configuration"
    elif search_model_name == "qfunction":
        configs = benchmark_bundle.q_function_nn_configs
        config_desc = "Q-function configuration"
    else:
        raise ValueError(f"Unknown search model name: {search_model_name}")

    if configs is None:
        raise click.UsageError(f"Benchmark '{benchmark_name}' does not define a {config_desc}.")

    config = configs.get(model_type)
    if config is None:
        raise click.UsageError(
            f"Benchmark '{benchmark_name}' {config_desc} type '{model_type}' not available."
        )

    path_template = config.param_path
    if path_template is None:
        raise click.UsageError(
            f"Benchmark '{benchmark_name}' {config_desc} '{model_type}' has no param path."
        )

    resolved_param_path = _resolve_param_path(path_template, puzzle, param_path)
    neural_config = {}
    if aqt_cfg:
        neural_config["aqt_cfg"] = aqt_cfg

    solver = config.callable(
        puzzle=puzzle,
        path=resolved_param_path,
        init_params=False,
        **neural_config,
    )
    attach_runtime_metadata(
        solver,
        model_type=model_type,
        param_path=resolved_param_path,
        extra={"aqt_cfg": aqt_cfg},
    )
    return solver


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


def _execute_benchmark_command(
    puzzle: Puzzle,
    eval_options: EvalOptions,
    benchmark,
    benchmark_name: str,
    benchmark_bundle,
    benchmark_cli_options,
    param_path: Optional[str],
    model_type: str,
    use_quantize: bool,
    quant_type: str,
    output_dir: Optional[Path],
    search_builder_fn,
    search_model_name: str,
    run_label: str,
    **kwargs,
):
    aqt_cfg = quant_type if use_quantize else None
    solver = _load_benchmark_solver(
        puzzle,
        benchmark_name,
        benchmark_bundle,
        param_path,
        model_type,
        aqt_cfg,
        search_model_name,
    )
    puzzle_opts = PuzzleOptions(puzzle=benchmark_name)
    _run_benchmark(
        puzzle=puzzle,
        puzzle_name=benchmark_name,
        search_model=solver,
        search_model_name=search_model_name,
        run_label=run_label,
        search_builder_fn=search_builder_fn,
        eval_options=eval_options,
        puzzle_opts=puzzle_opts,
        benchmark=benchmark,
        benchmark_name=benchmark_name,
        benchmark_bundle=benchmark_bundle,
        benchmark_cli_options=benchmark_cli_options,
        output_dir=output_dir,
    )


@benchmark.command(name="astar")
@benchmark_options
@eval_options
@_benchmark_model_options
def benchmark_astar(**kwargs):
    _execute_benchmark_command(
        search_builder_fn=astar_builder,
        search_model_name="heuristic",
        run_label="astar",
        **kwargs,
    )


@benchmark.command(name="astar-d")
@benchmark_options
@eval_options
@_benchmark_model_options
def benchmark_astar_d(**kwargs):
    _execute_benchmark_command(
        search_builder_fn=astar_d_builder,
        search_model_name="heuristic",
        run_label="astar_d",
        **kwargs,
    )


@benchmark.command(name="bi-astar")
@benchmark_options
@eval_options
@_benchmark_model_options
def benchmark_bi_astar(**kwargs):
    _execute_benchmark_command(
        search_builder_fn=bi_astar_builder,
        search_model_name="heuristic",
        run_label="bi_astar",
        **kwargs,
    )


@benchmark.command(name="bi-astar-d")
@benchmark_options
@eval_options
@_benchmark_model_options
def benchmark_bi_astar_d(**kwargs):
    _execute_benchmark_command(
        search_builder_fn=bi_astar_d_builder,
        search_model_name="heuristic",
        run_label="bi_astar_d",
        **kwargs,
    )


@benchmark.command(name="qstar")
@benchmark_options
@eval_options
@_benchmark_model_options
def benchmark_qstar(**kwargs):
    _execute_benchmark_command(
        search_builder_fn=qstar_builder,
        search_model_name="qfunction",
        run_label="qstar",
        **kwargs,
    )


@benchmark.command(name="bi-qstar")
@benchmark_options
@eval_options
@_benchmark_model_options
def benchmark_bi_qstar(**kwargs):
    _execute_benchmark_command(
        search_builder_fn=bi_qstar_builder,
        search_model_name="qfunction",
        run_label="bi_qstar",
        **kwargs,
    )


@benchmark.command(name="beam")
@benchmark_options
@eval_options(variant="beam")
@_benchmark_model_options
def benchmark_beam(**kwargs):
    _execute_benchmark_command(
        search_builder_fn=beam_builder,
        search_model_name="heuristic",
        run_label="beam",
        **kwargs,
    )


@benchmark.command(name="qbeam")
@benchmark_options
@eval_options(variant="beam")
@_benchmark_model_options
def benchmark_qbeam(**kwargs):
    _execute_benchmark_command(
        search_builder_fn=qbeam_builder,
        search_model_name="qfunction",
        run_label="qbeam",
        **kwargs,
    )


@benchmark.command(name="id-astar")
@benchmark_options
@eval_options
@_benchmark_model_options
def benchmark_id_astar(**kwargs):
    _execute_benchmark_command(
        search_builder_fn=id_astar_builder,
        search_model_name="heuristic",
        run_label="id_astar",
        **kwargs,
    )


@benchmark.command(name="id-qstar")
@benchmark_options
@eval_options
@_benchmark_model_options
def benchmark_id_qstar(**kwargs):
    _execute_benchmark_command(
        search_builder_fn=id_qstar_builder,
        search_model_name="qfunction",
        run_label="id_qstar",
        **kwargs,
    )


__all__ = [
    "benchmark",
    "benchmark_astar",
    "benchmark_astar_d",
    "benchmark_bi_astar",
    "benchmark_bi_astar_d",
    "benchmark_qstar",
    "benchmark_bi_qstar",
    "benchmark_beam",
    "benchmark_qbeam",
    "benchmark_id_astar",
    "benchmark_id_qstar",
]
