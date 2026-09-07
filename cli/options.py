"""Typed CLI inputs and explicit runtime configuration resolution."""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, Literal

import jax
import tyro

from config.pydantic_models import (
    DistTrainOptions,
    EvalOptions,
    HeuristicOptions,
    PuzzleConfig,
    PuzzleOptions,
    QFunctionOptions,
    SearchOptions,
    VisualizeOptions,
)
from helpers.formatting import human_format_to_float, parse_human_int
from helpers.param_stats import attach_runtime_metadata
from helpers.util import map_kwargs_to_pydantic
from heuristic.heuristic_base import Heuristic
from qfunction.q_base import QFunction
from train_util.optimizer import OPTIMIZERS


def parse_bool(value: str) -> bool:
    """Accept the existing CLI's case-insensitive boolean values."""
    if value.lower() in ("1", "true", "t", "yes", "y", "on"):
        return True
    if value.lower() in ("0", "false", "f", "no", "n", "off"):
        return False
    raise ValueError(f"{value!r} is not a valid boolean")


def _json_object(value: str, option: str) -> dict:
    parsed = json.loads(value)
    if not isinstance(parsed, dict):
        raise ValueError(f"{option} must be a JSON object.")
    return parsed


HumanInt = Annotated[
    int,
    tyro.constructors.PrimitiveConstructorSpec(
        nargs=1,
        metavar="INT",
        instance_from_str=lambda values: parse_human_int(values[0]),
        is_instance=lambda value: type(value) is int,
        str_from_instance=lambda value: [str(value)],
    ),
]
HumanFloat = Annotated[
    float,
    tyro.constructors.PrimitiveConstructorSpec(
        nargs=1,
        metavar="FLOAT",
        instance_from_str=lambda values: human_format_to_float(values[0]),
        is_instance=lambda value: isinstance(value, (int, float)) and not isinstance(value, bool),
        str_from_instance=lambda value: [str(value)],
    ),
]
BoolValue = Annotated[
    bool,
    tyro.constructors.PrimitiveConstructorSpec(
        nargs=1,
        metavar="BOOL",
        instance_from_str=lambda values: parse_bool(values[0]),
        is_instance=lambda value: type(value) is bool,
        str_from_instance=lambda value: [str(value)],
    ),
]


PuzzleName = Annotated[
    str,
    tyro.conf.arg(
        constructor_factory=lambda: tyro.extras.literal_type_from_choices(_puzzle_bundles())
    ),
]


@dataclass
class PuzzleArgs:
    puzzle: Annotated[
        PuzzleName, tyro.conf.arg(aliases=["-p"], help="Puzzle to solve")
    ] = "n-puzzle"
    hard: Annotated[bool, tyro.conf.arg(aliases=["-h"], help="Use the hard puzzle")] = False
    puzzle_args: Annotated[
        str, tyro.conf.arg(aliases=["-pargs"], help="Arguments for the puzzle")
    ] = ""
    seeds: Annotated[str, tyro.conf.arg(aliases=["-s"], help="Seed for the random puzzle")] = "0"


@dataclass
class TrainPuzzleArgs:
    puzzle: Annotated[
        PuzzleName, tyro.conf.arg(aliases=["-p"], help="Puzzle to solve")
    ] = "rubikscube"
    puzzle_args: Annotated[
        str, tyro.conf.arg(aliases=["-pargs"], help="Arguments for the puzzle")
    ] = ""


@dataclass
class BenchmarkArgs:
    benchmark_key: Annotated[
        str | None,
        tyro.conf.DisallowNone,
        tyro.conf.arg(
            name="benchmark",
            help="Exact benchmark dataset. Defaults to 'rubikscube-deepcubea' unless --puzzle is set.",
        ),
    ] = None
    puzzle_key: Annotated[
        PuzzleName | None,
        tyro.conf.DisallowNone,
        tyro.conf.arg(
            name="puzzle",
            aliases=["-p"],
            help="Generate benchmark samples from a puzzle without an exact dataset.",
        ),
    ] = None
    puzzle_args: Annotated[
        str, tyro.conf.arg(aliases=["-pargs"], help="Arguments for the generated puzzle.")
    ] = ""
    benchmark_args: Annotated[
        str, tyro.conf.arg(help="JSON string with keyword arguments for the benchmark constructor.")
    ] = ""
    sample_limit: Annotated[
        int | None,
        tyro.conf.DisallowNone,
        tyro.conf.arg(help="Maximum number of samples to evaluate from the benchmark dataset."),
    ] = None
    sample_ids: Annotated[
        str,
        tyro.conf.arg(
            help="Comma-separated list of sample IDs to evaluate. Overrides sample-limit when provided."
        ),
    ] = ""


@dataclass
class SearchArgs:
    max_node_size: Annotated[
        HumanInt | None,
        tyro.conf.DisallowNone,
        tyro.conf.arg(aliases=["-m"], help="Size of the puzzle"),
    ] = None
    batch_size: Annotated[
        HumanInt | None,
        tyro.conf.DisallowNone,
        tyro.conf.arg(aliases=["-b"], help="Batch size for BGPQ"),
    ] = None
    cost_weight: Annotated[
        HumanFloat | None,
        tyro.conf.DisallowNone,
        tyro.conf.arg(aliases=["-w"], help="Weight for the A* search"),
    ] = None
    pop_ratio: Annotated[
        HumanFloat | None,
        tyro.conf.DisallowNone,
        tyro.conf.arg(aliases=["-pr"], help="Ratio for popping nodes from the priority queue."),
    ] = None
    bound_step: Annotated[
        HumanFloat | None,
        tyro.conf.DisallowNone,
        tyro.conf.arg(help="ID{}* threshold grid size. 0 uses the exact next_bound ladder."),
    ] = None
    max_path_len: Annotated[
        HumanInt | None,
        tyro.conf.DisallowNone,
        tyro.conf.arg(
            help="ID{}* action-history length; also caps search depth. Dominates stack memory."
        ),
    ] = None
    vmap_size: Annotated[
        HumanInt | None,
        tyro.conf.DisallowNone,
        tyro.conf.arg(aliases=["-vm"], help="Size for the vmap"),
    ] = None
    debug: Annotated[bool | None, tyro.conf.DisallowNone, tyro.conf.arg(help="Debug mode")] = None
    profile: Annotated[
        bool | None, tyro.conf.DisallowNone, tyro.conf.arg(help="Profile mode")
    ] = None
    show_compile_time: Annotated[
        bool | None, tyro.conf.DisallowNone, tyro.conf.arg(help="Show compile time")
    ] = None
    emit_workload_signature: Annotated[
        bool | None,
        tyro.conf.DisallowNone,
        tyro.conf.arg(help="Emit Xtructure workload signature (xtr_* metrics)."),
    ] = None
    search_preset: Annotated[
        str | None, tyro.conf.DisallowNone, tyro.conf.arg(help="Name of the search preset to use.")
    ] = None


@dataclass
class EvalArgs:
    batch_size: Annotated[
        HumanInt | None,
        tyro.conf.DisallowNone,
        tyro.conf.arg(aliases=["-b"], help="Batch size for search."),
    ] = None
    show_compile_time: Annotated[
        bool | None, tyro.conf.DisallowNone, tyro.conf.arg(help="Show compile time")
    ] = None
    emit_workload_signature: Annotated[
        bool | None,
        tyro.conf.DisallowNone,
        tyro.conf.arg(help="Emit Xtructure workload signature (xtr_* metrics)."),
    ] = None
    max_node_size: Annotated[
        HumanInt | None,
        tyro.conf.DisallowNone,
        tyro.conf.arg(aliases=["-m"], help="Maximum number of nodes to search."),
    ] = None
    cost_weight: Annotated[
        HumanFloat | None,
        tyro.conf.DisallowNone,
        tyro.conf.arg(aliases=["-w"], help="Weight for cost in search."),
    ] = None
    pop_ratio: Annotated[
        str | None,
        tyro.conf.DisallowNone,
        tyro.conf.arg(
            aliases=["-pr"],
            help="Ratio(s) for popping nodes from the priority queue. Can be a single float, 'inf', or a comma-separated list (e.g., 'inf,0.4,0.3').",
        ),
    ] = None
    bound_step: Annotated[
        HumanFloat | None,
        tyro.conf.DisallowNone,
        tyro.conf.arg(help="ID{}* threshold grid size. 0 uses the exact next_bound ladder."),
    ] = None
    max_path_len: Annotated[
        HumanInt | None,
        tyro.conf.DisallowNone,
        tyro.conf.arg(
            help="ID{}* action-history length; also caps search depth. Dominates stack memory."
        ),
    ] = None
    num_eval: Annotated[
        HumanInt | None,
        tyro.conf.DisallowNone,
        tyro.conf.arg(aliases=["-ne"], help="Number of puzzles to evaluate."),
    ] = None
    run_name: Annotated[
        str | None,
        tyro.conf.DisallowNone,
        tyro.conf.arg(aliases=["-rn"], help="Name of the evaluation run."),
    ] = None
    use_early_stopping: Annotated[
        BoolValue | None,
        tyro.conf.DisallowNone,
        tyro.conf.arg(help="Enable early stopping based on success rate threshold."),
    ] = None
    early_stop_patience: Annotated[
        HumanInt | None,
        tyro.conf.DisallowNone,
        tyro.conf.arg(help="Number of samples to check before considering early stopping."),
    ] = None
    early_stop_threshold: Annotated[
        HumanFloat | None,
        tyro.conf.DisallowNone,
        tyro.conf.arg(help="Minimum success rate threshold for early stopping (0.0 to 1.0)."),
    ] = None
    eval_preset: Annotated[
        str | None,
        tyro.conf.DisallowNone,
        tyro.conf.arg(help="Name of the evaluation preset to use."),
    ] = None


@dataclass
class HeuristicArgs:
    neural_heuristic: Annotated[
        bool, tyro.conf.arg(aliases=["-nn"], help="Use neural heuristic")
    ] = False
    param_path: Annotated[
        str | None,
        tyro.conf.DisallowNone,
        tyro.conf.arg(help="Path to the heuristic parameter file."),
    ] = None
    model_type: Annotated[
        str | None, tyro.conf.DisallowNone, tyro.conf.arg(help="Type of the heuristic model.")
    ] = None
    use_quantize: Annotated[
        bool, tyro.conf.arg(aliases=["-q"], help="Use quantization (defaults to int8).")
    ] = False
    quant_type: Annotated[
        Literal["int8", "int4", "int4_w8a", "int8_w_only"],
        tyro.conf.arg(help="Specific AQT quantization configuration to use."),
    ] = "int8"


@dataclass
class QFunctionArgs:
    neural_qfunction: Annotated[
        bool, tyro.conf.arg(aliases=["-nn"], help="Use neural q function")
    ] = False
    param_path: Annotated[
        str | None,
        tyro.conf.DisallowNone,
        tyro.conf.arg(help="Path to the Q-function parameter file."),
    ] = None
    model_type: Annotated[
        str | None, tyro.conf.DisallowNone, tyro.conf.arg(help="Type of the Q-function model.")
    ] = None
    use_quantize: Annotated[
        bool, tyro.conf.arg(aliases=["-q"], help="Use quantization (defaults to int8).")
    ] = False
    quant_type: Annotated[
        Literal["int8", "int4", "int4_w8a", "int8_w_only"],
        tyro.conf.arg(help="Specific AQT quantization configuration to use."),
    ] = "int8"


@dataclass
class VisualizeArgs:
    visualize_terminal: Annotated[
        bool, tyro.conf.arg(aliases=["-vt"], help="Visualize the path with terminal")
    ] = False
    visualize_imgs: Annotated[
        bool, tyro.conf.arg(aliases=["-vi"], help="Visualize the path with gif images")
    ] = False
    max_animation_time: Annotated[
        int, tyro.conf.arg(aliases=["-mt"], help="Max animation time")
    ] = 10


@dataclass
class TrainArgs:
    steps: Annotated[HumanInt | None, tyro.conf.DisallowNone, tyro.conf.arg(aliases=["-s"])] = None
    dataset_batch_size: Annotated[
        HumanInt | None, tyro.conf.DisallowNone, tyro.conf.arg(aliases=["-db"])
    ] = None
    dataset_minibatch_size: Annotated[
        HumanInt | None, tyro.conf.DisallowNone, tyro.conf.arg(aliases=["-dmb"])
    ] = None
    sampling_non_backtracking_steps: Annotated[
        HumanInt | None,
        tyro.conf.DisallowNone,
        tyro.conf.arg(
            help="Number of previous states to avoid revisiting during dataset sampling."
        ),
    ] = None
    train_minibatch_size: Annotated[
        HumanInt | None, tyro.conf.DisallowNone, tyro.conf.arg(aliases=["-tmb"])
    ] = None
    key: Annotated[int | None, tyro.conf.DisallowNone, tyro.conf.arg(aliases=["-k"])] = None
    reset: Annotated[BoolValue | None, tyro.conf.DisallowNone, tyro.conf.arg(aliases=["-r"])] = None
    loss_threshold: Annotated[
        HumanFloat | None, tyro.conf.DisallowNone, tyro.conf.arg(aliases=["-lt"])
    ] = None
    update_interval: Annotated[
        HumanInt | None, tyro.conf.DisallowNone, tyro.conf.arg(aliases=["-ui"])
    ] = None
    force_update_interval: Annotated[
        HumanInt | None, tyro.conf.DisallowNone, tyro.conf.arg(aliases=["-fui"])
    ] = None
    use_soft_update: Annotated[
        bool | None, tyro.conf.DisallowNone, tyro.conf.arg(aliases=["-su"])
    ] = None
    use_double_dqn: Annotated[
        bool | None,
        tyro.conf.DisallowNone,
        tyro.conf.arg(aliases=["-ddn"], help="Enable Double DQN target computation."),
    ] = None
    using_hindsight_target: Annotated[
        bool | None, tyro.conf.DisallowNone, tyro.conf.arg(aliases=["-her"])
    ] = None
    using_triangular_sampling: Annotated[
        bool | None, tyro.conf.DisallowNone, tyro.conf.arg(aliases=["-ts"])
    ] = None
    label: Annotated[
        Literal["td", "diffusion", "warmup_td"] | None,
        tyro.conf.DisallowNone,
        tyro.conf.arg(
            help="Training target generation: 'td' bootstrap targets (DAVI / Q-learning) min-capped by diffusion trajectory distances, 'diffusion' trajectory Bellman propagation, or 'warmup_td' diffusion targets for the first --warmup_ratio of steps before switching to td. Default: 'td'."
        ),
    ] = None
    warmup_ratio: Annotated[
        HumanFloat | None,
        tyro.conf.DisallowNone,
        tyro.conf.arg(
            help="Fraction of training steps using diffusion targets with --label warmup_td."
        ),
    ] = None
    temperature: Annotated[
        HumanFloat | None,
        tyro.conf.DisallowNone,
        tyro.conf.arg(aliases=["-tp"], help="Boltzmann temperature for action selection."),
    ] = None
    debug: Annotated[bool | None, tyro.conf.DisallowNone, tyro.conf.arg(aliases=["-d"])] = None
    multi_device: Annotated[
        BoolValue | None, tyro.conf.DisallowNone, tyro.conf.arg(aliases=["-md"])
    ] = None
    opt_state_reset: Annotated[
        BoolValue | None, tyro.conf.DisallowNone, tyro.conf.arg(aliases=["-osr"])
    ] = None
    optimizer: Annotated[str, tyro.conf.arg(help="Optimizer to use")] = "muon"
    learning_rate: Annotated[
        HumanFloat | None, tyro.conf.DisallowNone, tyro.conf.arg(aliases=["-lr"])
    ] = None
    weight_decay_size: Annotated[
        HumanFloat | None,
        tyro.conf.DisallowNone,
        tyro.conf.arg(aliases=["-wd"], help="Weight decay size for regularization."),
    ] = None
    loss: Annotated[
        Literal["mse", "huber", "logcosh", "asymmetric_huber", "asymmetric_logcosh"] | None,
        tyro.conf.DisallowNone,
        tyro.conf.arg(help="Select training loss."),
    ] = None
    loss_args: Annotated[
        str | None,
        tyro.conf.DisallowNone,
        tyro.conf.arg(
            help='JSON object of additional keyword arguments for the selected loss, e.g. \'{"huber_delta":0.2,"asymmetric_tau":0.1}\'.'
        ),
    ] = None
    eval_count: Annotated[
        HumanInt | None,
        tyro.conf.DisallowNone,
        tyro.conf.arg(
            aliases=["-ec"], help="Number of evaluations to perform during training (default: 5)."
        ),
    ] = None
    eval_search_metric: Annotated[
        Literal["astar", "astar_d", "bi_astar", "bi_astar_d", "beam", "qstar", "bi_qstar", "qbeam"]
        | None,
        tyro.conf.DisallowNone,
        tyro.conf.arg(
            help="Search algorithm to use for evaluation during training (heuristic: astar/astar_d/bi_astar/bi_astar_d/beam, qfunction: qstar/bi_qstar/qbeam)."
        ),
    ] = None
    k_max: Annotated[
        HumanInt | None,
        tyro.conf.DisallowNone,
        tyro.conf.arg(
            aliases=["-km"], help="Override puzzle's default k_max (formerly shuffle_length)."
        ),
    ] = None
    logger: Annotated[
        Literal["aim", "tensorboard", "wandb", "none"] | None,
        tyro.conf.DisallowNone,
        tyro.conf.arg(help="Logger to use."),
    ] = None
    preset: Annotated[
        str,
        tyro.conf.arg(aliases=["-pre"], help="Training configuration preset for heuristic train."),
    ] = "davi"


@dataclass
class NeuralArgs:
    param_path: Annotated[
        str | None,
        tyro.conf.DisallowNone,
        tyro.conf.arg(help="Path to the heuristic parameter file."),
    ] = None
    neural_config: Annotated[
        str | None,
        tyro.conf.DisallowNone,
        tyro.conf.arg(
            aliases=["-nc"], help="Neural configuration. Overrides the default configuration."
        ),
    ] = None
    model_type: Annotated[
        str | None, tyro.conf.DisallowNone, tyro.conf.arg(help="Type of the heuristic model.")
    ] = None
    use_quantize: Annotated[
        bool, tyro.conf.arg(aliases=["-q"], help="Use quantization (defaults to int8).")
    ] = False
    quant_type: Annotated[
        Literal["int8", "int4", "int4_w8a", "int8_w_only"],
        tyro.conf.arg(help="Specific AQT quantization configuration to use."),
    ] = "int8"


@dataclass
class HeuristicSearchArgs(PuzzleArgs, SearchArgs, VisualizeArgs, HeuristicArgs):
    """Puzzle, search, visualization and heuristic settings."""


@dataclass
class QFunctionSearchArgs(PuzzleArgs, SearchArgs, VisualizeArgs, QFunctionArgs):
    """Puzzle, search, visualization and Q-function settings."""


@dataclass
class HeuristicBenchmarkArgs(BenchmarkArgs, EvalArgs, HeuristicArgs):
    output_dir: Path | None = None


@dataclass
class QFunctionBenchmarkArgs(BenchmarkArgs, EvalArgs, QFunctionArgs):
    output_dir: Path | None = None


@dataclass
class HeuristicTrainArgs(TrainPuzzleArgs, TrainArgs, NeuralArgs):
    """Train a neural heuristic. Unspecified values come from the preset."""


@dataclass
class QFunctionTrainArgs(TrainPuzzleArgs, TrainArgs, NeuralArgs):
    """Train a neural Q-function. Unspecified values come from the preset."""

    preset: Annotated[str, tyro.conf.arg(aliases=["-pre"])] = "qlearning"


def _puzzle_bundles():
    from config import puzzle_bundles

    return puzzle_bundles


def _benchmark_bundles():
    from config import benchmark_bundles

    return benchmark_bundles


def _train_presets():
    from config import train_presets

    return train_presets


def _setup_neural_component(
    puzzle_bundle,
    puzzle,
    puzzle_name,
    component_type,
    param_path,
    neural_config_override,
    reset_params,
    model_type="default",
    aqt_cfg=None,
):
    if component_type == "heuristic":
        nn_configs = puzzle_bundle.heuristic_nn_configs
        config_key = "heuristic_config"
        comp_key = "heuristic"
        err_msg = "Neural heuristic"
    else:
        nn_configs = puzzle_bundle.q_function_nn_configs
        config_key = "q_config"
        comp_key = "qfunction"
        err_msg = "Neural Q-function"

    if nn_configs is None:
        raise ValueError(f"{err_msg} not available for puzzle '{puzzle_name}'.")

    nn_config = nn_configs.get(model_type)
    if nn_config is None:
        raise ValueError(
            f"{err_msg} config type '{model_type}' not available for puzzle '{puzzle_name}'."
        )

    if param_path is None:
        path_template = nn_config.param_path
        if path_template is None:
            raise ValueError(f"Default parameter path not found for puzzle '{puzzle_name}'.")
        if "{size}" in path_template:
            param_path = path_template.format(size=puzzle.size)
        else:
            param_path = path_template

    final_neural_config = {}
    if neural_config_override is not None:
        final_neural_config.update(_json_object(neural_config_override, "--neural-config"))

    if aqt_cfg is not None:
        final_neural_config["aqt_cfg"] = aqt_cfg

    component = nn_config.callable(
        puzzle=puzzle,
        path=param_path,
        init_params=reset_params,
        **final_neural_config,
    )
    # Attach runtime metadata (model type / path / param stats) for nicer config printing.
    attach_runtime_metadata(
        component,
        model_type=model_type,
        param_path=param_path,
        extra={"cli_neural_config": final_neural_config},
    )
    return {comp_key: component, config_key: final_neural_config}


def _build_puzzle(puzzle_opts: PuzzleOptions, *, default_hard: bool):
    puzzle_name = puzzle_opts.puzzle
    bundles = _puzzle_bundles()
    if puzzle_name not in bundles:
        raise ValueError(f"Unknown puzzle '{puzzle_name}'. Available: {list(bundles)}")
    puzzle_bundle = bundles[puzzle_name]

    input_args = (
        _json_object(puzzle_opts.puzzle_args, "--puzzle-args") if puzzle_opts.puzzle_args else {}
    )
    puzzle_opts.hard = default_hard or puzzle_opts.hard
    if puzzle_opts.hard and puzzle_bundle.puzzle_hard is not None:
        puzzle_callable = puzzle_bundle.puzzle_hard
    else:
        puzzle_callable = puzzle_bundle.puzzle

    if isinstance(puzzle_callable, PuzzleConfig):
        puzzle_kwargs = {**puzzle_callable.kwargs, **input_args}
        if puzzle_callable.initial_shuffle is not None and "initial_shuffle" not in puzzle_kwargs:
            puzzle_kwargs["initial_shuffle"] = puzzle_callable.initial_shuffle
        puzzle_instance = puzzle_callable.callable(**puzzle_kwargs)
    elif puzzle_callable is None:
        raise ValueError(
            f"Puzzle type for '{puzzle_name}'"
            f"{' (hard)' if puzzle_opts.hard else ''} is not defined."
        )
    else:
        puzzle_instance = puzzle_callable(**input_args)

    return puzzle_name, puzzle_bundle, puzzle_instance


def resolve_puzzle(
    kwargs: dict, *, default_hard: bool = False, use_seeds_flag: bool = True
) -> dict:
    puzzle_kwargs = map_kwargs_to_pydantic(PuzzleOptions, kwargs)
    puzzle_opts = PuzzleOptions(**puzzle_kwargs)
    puzzle_name, puzzle_bundle, puzzle_instance = _build_puzzle(
        puzzle_opts, default_hard=default_hard
    )
    kwargs["puzzle"] = puzzle_instance
    kwargs["puzzle_name"] = puzzle_name
    kwargs["puzzle_bundle"] = puzzle_bundle
    if use_seeds_flag:
        kwargs["seeds"] = puzzle_opts.get_seed_list()
    kwargs["puzzle_opts"] = puzzle_opts
    return kwargs


def resolve_benchmark(kwargs: dict) -> dict:
    bundles = _benchmark_bundles()
    if not bundles:
        raise ValueError("No benchmark bundles registered.")
    default_benchmark = next(iter(bundles))
    benchmark_key = kwargs.pop("benchmark_key")
    puzzle_key = kwargs.pop("puzzle_key")
    puzzle_args = kwargs.pop("puzzle_args")
    benchmark_args_override = kwargs.pop("benchmark_args")
    sample_ids_raw = kwargs.pop("sample_ids")
    sample_limit = kwargs.pop("sample_limit")
    if benchmark_key and puzzle_key:
        raise ValueError("Use either --benchmark or --puzzle, not both.")
    if puzzle_args and (not puzzle_key):
        raise ValueError("--puzzle-args requires --puzzle.")
    if puzzle_key:
        if benchmark_args_override or sample_ids_raw or sample_limit is not None:
            raise ValueError(
                "--benchmark-args, --sample-ids, and --sample-limit require --benchmark."
            )
        puzzle_opts = PuzzleOptions(puzzle=puzzle_key, puzzle_args=puzzle_args)
        puzzle_name, puzzle_bundle, puzzle_instance = _build_puzzle(puzzle_opts, default_hard=True)
        kwargs.update(
            benchmark=None,
            benchmark_name=None,
            benchmark_bundle=None,
            benchmark_cli_options={},
            puzzle=puzzle_instance,
            puzzle_name=puzzle_name,
            puzzle_bundle=puzzle_bundle,
            puzzle_opts=puzzle_opts,
        )
        return kwargs
    benchmark_key = benchmark_key or default_benchmark
    if benchmark_key not in bundles:
        raise ValueError(f"Unknown benchmark '{benchmark_key}'. Available: {list(bundles)}")
    benchmark_bundle = bundles[benchmark_key]
    benchmark_args = dict(benchmark_bundle.benchmark_args or {})
    if benchmark_args_override:
        try:
            benchmark_args.update(_json_object(benchmark_args_override, "--benchmark-args"))
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid JSON provided to --benchmark-args: {exc}") from exc
    benchmark_instance = benchmark_bundle.benchmark(**benchmark_args)
    sample_ids = None
    if sample_ids_raw:
        try:
            sample_ids = [
                int(part.strip()) for part in sample_ids_raw.split(",") if part.strip() != ""
            ]
        except ValueError as exc:
            raise ValueError(
                "Invalid value in --sample-ids. Expected comma-separated integers."
            ) from exc
    kwargs["benchmark"] = benchmark_instance
    kwargs["benchmark_name"] = benchmark_key
    kwargs["benchmark_bundle"] = benchmark_bundle
    kwargs["benchmark_cli_options"] = {"sample_limit": sample_limit, "sample_ids": sample_ids}
    kwargs["puzzle"] = benchmark_instance.puzzle
    kwargs["puzzle_name"] = benchmark_key
    kwargs["puzzle_bundle"] = benchmark_bundle
    kwargs["puzzle_opts"] = PuzzleOptions(puzzle=benchmark_key)
    return kwargs


def resolve_search(kwargs: dict, *, variant: str = "default") -> dict:
    overrides = map_kwargs_to_pydantic(SearchOptions, kwargs)
    puzzle_bundle = kwargs["puzzle_bundle"]
    search_preset = kwargs.pop("search_preset")
    search_options_configs = getattr(puzzle_bundle, "search_options_configs", None)
    if search_options_configs:
        preset_key = search_preset or ("beam" if variant == "beam" else "default")
        if preset_key in search_options_configs:
            base_search_options = search_options_configs[preset_key]
        elif search_preset:
            puzzle_name = kwargs.get("puzzle_name") or "unknown"
            raise ValueError(
                f"Search preset '{search_preset}' not available for '{puzzle_name}'. Available: {list(search_options_configs.keys())}"
            )
        else:
            base_search_options = SearchOptions()
    else:
        base_search_options = SearchOptions()
    search_opts = base_search_options.model_copy(update=overrides)
    if search_opts.debug:
        print("Disabling JIT")
        import jax

        jax.config.update("jax_disable_jit", True)
        search_opts.max_node_size = 10000
        search_opts.batch_size = 100
    kwargs["search_options"] = search_opts
    return kwargs


def resolve_eval(kwargs: dict, *, variant: str = "default") -> dict:
    overrides = map_kwargs_to_pydantic(EvalOptions, kwargs)
    puzzle_bundle = kwargs["puzzle_bundle"]
    eval_preset = kwargs.pop("eval_preset")
    eval_options_configs = getattr(puzzle_bundle, "eval_options_configs", None)
    if eval_options_configs:
        preset_key = eval_preset or ("beam" if variant == "beam" else "default")
        if preset_key in eval_options_configs:
            base_eval_options = eval_options_configs[preset_key]
        elif eval_preset:
            puzzle_name = kwargs.get("puzzle_name") or kwargs.get("benchmark_name") or "unknown"
            raise ValueError(
                f"Eval preset '{eval_preset}' not available for '{puzzle_name}'. Available: {list(eval_options_configs.keys())}"
            )
        else:
            base_eval_options = EvalOptions()
    else:
        base_eval_options = EvalOptions()
    if "pop_ratio" in overrides and overrides["pop_ratio"] is not None:
        pop_ratio_val = overrides["pop_ratio"]
        pop_ratio_str = str(pop_ratio_val)
        if "," in pop_ratio_str:
            pop_ratios = []
            for pr_val in pop_ratio_str.split(","):
                try:
                    pop_ratios.append(float(pr_val.strip()))
                except ValueError as e:
                    if pr_val.strip().lower() == "inf":
                        pop_ratios.append(float("inf"))
                    else:
                        raise ValueError(f"Invalid pop_ratio value: {pr_val}") from e
            overrides["pop_ratio"] = pop_ratios
        else:
            try:
                overrides["pop_ratio"] = float(pop_ratio_str.strip())
            except ValueError as e:
                if pop_ratio_str.strip().lower() == "inf":
                    overrides["pop_ratio"] = float("inf")
                else:
                    raise ValueError(f"Invalid pop_ratio value: {pop_ratio_str}") from e
    eval_opts = base_eval_options.model_copy(update=overrides).resolve_for_eval_setup(
        has_benchmark=kwargs.get("benchmark") is not None
    )
    kwargs["eval_options"] = eval_opts
    return kwargs


def resolve_heuristic(kwargs: dict) -> dict:
    heuristic_kwargs = map_kwargs_to_pydantic(HeuristicOptions, kwargs)
    heuristic_opts = HeuristicOptions(**heuristic_kwargs)
    use_quantize = kwargs.pop("use_quantize")
    quant_type = kwargs.pop("quant_type")
    aqt_cfg = quant_type if use_quantize else None
    puzzle_bundle = kwargs.pop("puzzle_bundle")
    puzzle = kwargs["puzzle"]
    is_eval = kwargs.get("eval_options", None) is not None
    if heuristic_opts.neural_heuristic or is_eval:
        heuristic_configs = puzzle_bundle.heuristic_nn_configs
        if heuristic_configs is None:
            raise ValueError(
                f"Neural heuristic not available for puzzle '{kwargs['puzzle_name']}'."
            )
        model_type = heuristic_opts.model_type or "default"
        heuristic_config = heuristic_configs.get(model_type)
        if heuristic_config is None:
            raise ValueError(f"Neural heuristic config '{model_type}' not available.")
        param_path = heuristic_opts.param_path
        if param_path is None:
            path_template = heuristic_config.param_path
            if path_template is None:
                raise ValueError(f"Parameter path for type '{model_type}' not found.")
            if "{size}" in path_template:
                param_path = path_template.format(size=puzzle.size)
            else:
                param_path = path_template
        heuristic: Heuristic = heuristic_config.callable(
            puzzle=puzzle, path=param_path, init_params=False, aqt_cfg=aqt_cfg
        )
        attach_runtime_metadata(
            heuristic, model_type=model_type, param_path=param_path, extra={"aqt_cfg": aqt_cfg}
        )
    else:
        heuristic_callable = puzzle_bundle.heuristic
        if heuristic_callable is None:
            raise ValueError(f"Heuristic not available for puzzle '{kwargs['puzzle_name']}'.")
        heuristic: Heuristic = heuristic_callable(puzzle)
    kwargs["heuristic"] = heuristic
    kwargs["heuristic_options"] = heuristic_opts
    return kwargs


def resolve_qfunction(kwargs: dict) -> dict:
    q_kwargs = map_kwargs_to_pydantic(QFunctionOptions, kwargs)
    q_opts = QFunctionOptions(**q_kwargs)
    use_quantize = kwargs.pop("use_quantize")
    quant_type = kwargs.pop("quant_type")
    aqt_cfg = quant_type if use_quantize else None
    puzzle_bundle = kwargs.pop("puzzle_bundle")
    puzzle = kwargs["puzzle"]
    is_eval = kwargs.get("eval_options", None) is not None
    if q_opts.neural_qfunction or is_eval:
        q_configs = puzzle_bundle.q_function_nn_configs
        if q_configs is None:
            raise ValueError(
                f"Neural Q-function not available for puzzle '{kwargs['puzzle_name']}'."
            )
        model_type = q_opts.model_type or "default"
        q_config = q_configs.get(model_type)
        if q_config is None:
            raise ValueError(f"Neural Q-function config '{model_type}' not available.")
        param_path = q_opts.param_path
        if param_path is None:
            path_template = q_config.param_path
            if path_template is None:
                raise ValueError(f"Parameter path for type '{model_type}' not found.")
            if "{size}" in path_template:
                param_path = path_template.format(size=puzzle.size)
            else:
                param_path = path_template
        qfunction: QFunction = q_config.callable(
            puzzle=puzzle, path=param_path, init_params=False, aqt_cfg=aqt_cfg
        )
        attach_runtime_metadata(
            qfunction, model_type=model_type, param_path=param_path, extra={"aqt_cfg": aqt_cfg}
        )
    else:
        q_callable = puzzle_bundle.q_function
        if q_callable is None:
            raise ValueError(f"Q-function not available for puzzle '{kwargs['puzzle_name']}'.")
        qfunction: QFunction = q_callable(puzzle)
    kwargs["qfunction"] = qfunction
    kwargs["q_options"] = q_opts
    return kwargs


def resolve_visualize(kwargs: dict) -> dict:
    vis_kwargs = map_kwargs_to_pydantic(VisualizeOptions, kwargs)
    vis_opts = VisualizeOptions(**vis_kwargs)
    kwargs["visualize_options"] = vis_opts
    return kwargs


def resolve_train(kwargs: dict, *, preset_category: str) -> dict:
    preset_map = _train_presets()[preset_category]
    puzzle_bundle = kwargs["puzzle_bundle"]
    user_kmax = kwargs.pop("k_max")
    final_kmax = user_kmax if user_kmax is not None else puzzle_bundle.k_max
    kwargs["k_max"] = final_kmax
    preset_name = kwargs.pop("preset")
    if preset_name not in preset_map:
        raise ValueError(f"Unknown training preset '{preset_name}'. Available: {list(preset_map)}")
    preset = preset_map[preset_name]
    if kwargs.get("optimizer") is not None and kwargs["optimizer"] not in OPTIMIZERS:
        raise ValueError(
            f"Unknown optimizer '{kwargs['optimizer']}'. Available: {list(OPTIMIZERS)}"
        )
    overrides = map_kwargs_to_pydantic(DistTrainOptions, kwargs)
    if "eval_options" in overrides:
        cli_eval = overrides["eval_options"]
        if hasattr(cli_eval, "num_eval") and cli_eval.num_eval == -1:
            preset_eval_opts = getattr(preset, "eval_options", None)
            if preset_eval_opts:
                cli_eval.num_eval = preset_eval_opts.num_eval
        overrides["eval_options"] = cli_eval
    for key in list(kwargs.keys()):
        if key in DistTrainOptions.model_fields and kwargs[key] is None:
            kwargs.pop(key)
    if "loss_args" in overrides and isinstance(overrides["loss_args"], str):
        overrides["loss_args"] = _json_object(overrides["loss_args"], "--loss-args")
    train_opts = preset.model_copy(update=overrides)
    if train_opts.debug:
        print("Disabling JIT")
        jax.config.update("jax_disable_jit", True)
    kwargs["train_options"] = train_opts
    return kwargs


def resolve_train_heuristic(kwargs: dict) -> dict:
    puzzle_bundle = kwargs["puzzle_bundle"]
    puzzle = kwargs["puzzle"]
    puzzle_name = kwargs["puzzle_name"]
    reset = kwargs["train_options"].reset
    use_quantize = kwargs.pop("use_quantize")
    quant_type = kwargs.pop("quant_type")
    aqt_cfg = quant_type if use_quantize else None
    result = _setup_neural_component(
        puzzle_bundle,
        puzzle,
        puzzle_name,
        "heuristic",
        kwargs.pop("param_path"),
        kwargs.pop("neural_config"),
        reset,
        kwargs.pop("model_type") or "default",
        aqt_cfg=aqt_cfg,
    )
    kwargs.update(result)
    return kwargs


def resolve_train_qfunction(kwargs: dict) -> dict:
    puzzle_bundle = kwargs["puzzle_bundle"]
    puzzle = kwargs["puzzle"]
    puzzle_name = kwargs["puzzle_name"]
    reset = kwargs["train_options"].reset
    use_quantize = kwargs.pop("use_quantize")
    quant_type = kwargs.pop("quant_type")
    aqt_cfg = quant_type if use_quantize else None
    result = _setup_neural_component(
        puzzle_bundle,
        puzzle,
        puzzle_name,
        "q_function",
        kwargs.pop("param_path"),
        kwargs.pop("neural_config"),
        reset,
        kwargs.pop("model_type") or "default",
        aqt_cfg=aqt_cfg,
    )
    kwargs.update(result)
    return kwargs
