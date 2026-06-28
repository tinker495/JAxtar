"""Benchmark Click commands generated from the Search Algorithm Catalog.

See CONTEXT.md "Search Algorithm Catalog". The 10 algorithm-specific benchmark
subcommands (`benchmark astar`, `benchmark astar-d`, ...) are built by
iterating `SEARCH_ALGORITHM_CATALOG` and attaching to the `benchmark` Click
group.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

import click
from puxle import Puzzle

from config.algorithm_registry import SEARCH_ALGORITHM_CATALOG, SearchAlgorithmEntry
from config.pydantic_models import EvalOptions, PuzzleOptions
from helpers.param_stats import attach_runtime_metadata

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


def _build_benchmark_command(entry: SearchAlgorithmEntry) -> click.Command:
    eval_dec = eval_options(variant="beam") if entry.is_beam else eval_options

    def inner(**kwargs):
        _execute_benchmark_command(
            search_builder_fn=entry.builder_fn,
            search_model_name=entry.component_kind,
            run_label=entry.python_id,
            **kwargs,
        )

    inner = _benchmark_model_options(inner)
    inner = eval_dec(inner)
    inner = benchmark_options(inner)
    return click.command(name=entry.cli_subcommand)(inner)


for _entry in SEARCH_ALGORITHM_CATALOG:
    benchmark.add_command(_build_benchmark_command(_entry))


__all__ = ["benchmark"]
