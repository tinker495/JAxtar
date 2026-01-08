import json
from functools import wraps

import click
import jax

from config import benchmark_bundles, puzzle_bundles, train_presets, world_model_bundles
from config.pydantic_models import (
    DistTrainOptions,
    EvalOptions,
    HeuristicOptions,
    PuzzleConfig,
    PuzzleOptions,
    QFunctionOptions,
    SearchOptions,
    VisualizeOptions,
    WMDatasetOptions,
    WMGetDSOptions,
    WMGetModelOptions,
    WMTrainOptions,
    WorldModelPuzzleConfig,
)
from helpers.formatting import HUMAN_FLOAT, HUMAN_INT
from helpers.param_stats import attach_runtime_metadata
from helpers.util import map_kwargs_to_pydantic
from heuristic.heuristic_base import Heuristic
from qfunction.q_base import QFunction
from train_util.optimizer import OPTIMIZERS


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
        raise click.UsageError(f"{err_msg} not available for puzzle '{puzzle_name}'.")

    nn_config = nn_configs.get(model_type)
    if nn_config is None:
        raise click.UsageError(
            f"{err_msg} config type '{model_type}' not available for puzzle '{puzzle_name}'."
        )

    if param_path is None:
        path_template = nn_config.param_path
        if path_template is None:
            raise click.UsageError(f"Default parameter path not found for puzzle '{puzzle_name}'.")
        if "{size}" in path_template:
            param_path = path_template.format(size=puzzle.size)
        else:
            param_path = path_template

    final_neural_config = {}
    if neural_config_override is not None:
        final_neural_config.update(json.loads(neural_config_override))

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


def create_puzzle_options(
    default_puzzle: str,
    default_hard=False,
    use_hard_flag=False,
    puzzle_ds_flag=False,
    use_seeds_flag=False,
):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            puzzle_kwargs = map_kwargs_to_pydantic(PuzzleOptions, kwargs)
            puzzle_opts = PuzzleOptions(**puzzle_kwargs)

            puzzle_name = puzzle_opts.puzzle
            puzzle_bundle = puzzle_bundles[puzzle_name]

            input_args = {}
            if puzzle_opts.puzzle_args:
                input_args = json.loads(puzzle_opts.puzzle_args)

            puzzle_opts.hard = default_hard or puzzle_opts.hard
            if puzzle_opts.hard and puzzle_bundle.puzzle_hard is not None:
                puzzle_callable = puzzle_bundle.puzzle_hard
            elif puzzle_ds_flag:
                puzzle_callable = puzzle_bundle.puzzle
            else:
                puzzle_callable = puzzle_bundle.puzzle

            if isinstance(puzzle_callable, WorldModelPuzzleConfig):
                puzzle_instance = puzzle_callable.callable(path=puzzle_callable.path, **input_args)
            elif isinstance(puzzle_callable, PuzzleConfig):
                puzzle_kwargs = {**puzzle_callable.kwargs, **input_args}
                if (
                    puzzle_callable.initial_shuffle is not None
                    and "initial_shuffle" not in puzzle_kwargs
                ):
                    puzzle_kwargs["initial_shuffle"] = puzzle_callable.initial_shuffle
                puzzle_instance = puzzle_callable.callable(**puzzle_kwargs)
            elif puzzle_callable is None:
                raise click.UsageError(
                    f"Puzzle type for '{puzzle_name}'"
                    f"{' (hard)' if puzzle_opts.hard else ''} is not defined."
                )
            else:
                puzzle_instance = puzzle_callable(**input_args)

            kwargs["puzzle"] = puzzle_instance
            kwargs["puzzle_name"] = puzzle_name
            kwargs["puzzle_bundle"] = puzzle_bundle

            if use_seeds_flag:
                kwargs["seeds"] = puzzle_opts.get_seed_list()

            kwargs["puzzle_opts"] = puzzle_opts
            return func(*args, **kwargs)

        if use_seeds_flag:
            wrapper = click.option(
                "-s", "--seeds", default="0", type=str, help="Seed for the random puzzle"
            )(wrapper)

        wrapper = click.option(
            "-pargs", "--puzzle_args", default="", type=str, help="Arguments for the puzzle"
        )(wrapper)

        if use_hard_flag:
            wrapper = click.option(
                "-h", "--hard", default=False, is_flag=True, help="Use the hard puzzle"
            )(wrapper)

        if puzzle_ds_flag:
            choices = list(world_model_bundles.keys())
        else:
            choices = list(puzzle_bundles.keys())

        wrapper = click.option(
            "-p",
            "--puzzle",
            default=default_puzzle,
            type=click.Choice(choices),
            help="Puzzle to solve",
        )(wrapper)
        return wrapper

    return decorator


def benchmark_options(func: callable) -> callable:
    if not benchmark_bundles:
        raise RuntimeError("No benchmark bundles registered.")

    default_benchmark = next(iter(benchmark_bundles))

    @click.option(
        "--benchmark",
        "benchmark_key",
        default=default_benchmark,
        type=click.Choice(list(benchmark_bundles.keys())),
        help="Benchmark dataset to evaluate.",
    )
    @click.option(
        "--benchmark-args",
        default="",
        type=str,
        help="JSON string with keyword arguments for the benchmark constructor.",
    )
    @click.option(
        "--sample-limit",
        type=int,
        default=None,
        help="Maximum number of samples to evaluate from the benchmark dataset.",
    )
    @click.option(
        "--sample-ids",
        default="",
        type=str,
        help="Comma-separated list of sample IDs to evaluate. Overrides sample-limit when provided.",
    )
    @wraps(func)
    def wrapper(*args, **kwargs):
        benchmark_key = kwargs.pop("benchmark_key")
        benchmark_bundle = benchmark_bundles[benchmark_key]

        benchmark_args = dict(benchmark_bundle.benchmark_args or {})
        benchmark_args_override = kwargs.pop("benchmark_args")
        if benchmark_args_override:
            try:
                benchmark_args.update(json.loads(benchmark_args_override))
            except json.JSONDecodeError as exc:
                raise click.BadParameter(
                    f"Invalid JSON provided to --benchmark-args: {exc}"
                ) from exc

        benchmark_instance = benchmark_bundle.benchmark(**benchmark_args)

        sample_ids_raw = kwargs.pop("sample_ids")
        sample_ids = None
        if sample_ids_raw:
            try:
                sample_ids = [
                    int(part.strip()) for part in sample_ids_raw.split(",") if part.strip() != ""
                ]
            except ValueError as exc:
                raise click.BadParameter(
                    "Invalid value in --sample-ids. Expected comma-separated integers."
                ) from exc

        sample_limit = kwargs.pop("sample_limit")

        kwargs["benchmark"] = benchmark_instance
        kwargs["benchmark_name"] = benchmark_key
        kwargs["benchmark_bundle"] = benchmark_bundle
        kwargs["benchmark_cli_options"] = {
            "sample_limit": sample_limit,
            "sample_ids": sample_ids,
        }
        kwargs["puzzle"] = benchmark_instance.puzzle
        if "puzzle_bundle" not in kwargs:
            kwargs["puzzle_bundle"] = benchmark_bundle
        return func(*args, **kwargs)

    return wrapper


puzzle_options = create_puzzle_options(
    default_puzzle="n-puzzle", use_hard_flag=True, use_seeds_flag=True
)
eval_puzzle_options = create_puzzle_options(default_puzzle="rubikscube", default_hard=True)
dist_puzzle_options = create_puzzle_options(default_puzzle="rubikscube", default_hard=True)
wm_puzzle_ds_options = create_puzzle_options(default_puzzle="rubikscube", puzzle_ds_flag=True)


def search_options(func=None, *, variant: str = "default") -> callable:
    def decorator(func: callable) -> callable:
        @click.option(
            "-m", "--max_node_size", default=None, type=HUMAN_INT, help="Size of the puzzle"
        )
        @click.option(
            "-b", "--batch_size", default=None, type=HUMAN_INT, help="Batch size for BGPQ"
        )
        @click.option(
            "-w", "--cost_weight", default=None, type=HUMAN_FLOAT, help="Weight for the A* search"
        )
        @click.option(
            "-pr",
            "--pop_ratio",
            default=None,
            type=HUMAN_FLOAT,
            help="Ratio for popping nodes from the priority queue.",
        )
        @click.option("-vm", "--vmap_size", default=None, type=HUMAN_INT, help="Size for the vmap")
        @click.option("--debug", is_flag=True, default=None, help="Debug mode")
        @click.option("--profile", is_flag=True, default=None, help="Profile mode")
        @click.option("--show_compile_time", is_flag=True, default=None, help="Show compile time")
        @wraps(func)
        def wrapper(*args, **kwargs):
            overrides = map_kwargs_to_pydantic(SearchOptions, kwargs)

            puzzle_bundle = kwargs["puzzle_bundle"]
            if variant == "beam":
                base_search_options = puzzle_bundle.beam_search_options
            else:
                base_search_options = puzzle_bundle.search_options

            search_opts = base_search_options.model_copy(update=overrides)

            if search_opts.debug:
                print("Disabling JIT")
                jax.config.update("jax_disable_jit", True)
                search_opts.max_node_size = 10000
                search_opts.batch_size = 100

            kwargs["search_options"] = search_opts

            return func(*args, **kwargs)

        return wrapper

    if func is None:
        return decorator
    return decorator(func)


def eval_options(func=None, *, variant: str = "default") -> callable:
    def decorator(func: callable) -> callable:
        @click.option(
            "-b", "--batch-size", type=HUMAN_INT, default=None, help="Batch size for search."
        )
        @click.option("--show_compile_time", is_flag=True, default=None, help="Show compile time")
        @click.option(
            "-m",
            "--max-node-size",
            type=HUMAN_INT,
            default=None,
            help="Maximum number of nodes to search.",
        )
        @click.option(
            "-w", "--cost-weight", type=HUMAN_FLOAT, default=None, help="Weight for cost in search."
        )
        @click.option(
            "-pr",
            "--pop_ratio",
            type=str,
            default=None,
            help="Ratio(s) for popping nodes from the priority queue. Can be a single float, "
            "'inf', or a comma-separated list (e.g., 'inf,0.4,0.3').",
        )
        @click.option(
            "-ne", "--num-eval", type=HUMAN_INT, default=None, help="Number of puzzles to evaluate."
        )
        @click.option(
            "-rn", "--run-name", type=str, default=None, help="Name of the evaluation run."
        )
        @click.option(
            "--use-early-stopping",
            type=bool,
            default=None,
            help="Enable early stopping based on success rate threshold.",
        )
        @click.option(
            "--early-stop-patience",
            type=HUMAN_INT,
            default=None,
            help="Number of samples to check before considering early stopping.",
        )
        @click.option(
            "--early-stop-threshold",
            type=HUMAN_FLOAT,
            default=None,
            help="Minimum success rate threshold for early stopping (0.0 to 1.0).",
        )
        @wraps(func)
        def wrapper(*args, **kwargs):
            overrides = map_kwargs_to_pydantic(EvalOptions, kwargs)

            puzzle_bundle = kwargs["puzzle_bundle"]
            if variant == "beam":
                base_eval_options = getattr(puzzle_bundle, "beam_eval_options", None)
                if base_eval_options is None:
                    base_eval_options = puzzle_bundle.eval_options
            else:
                base_eval_options = puzzle_bundle.eval_options

            # pop_ratio special handling was done before map_kwargs_to_pydantic?
            # No, pop_ratio is in EvalOptions fields, so it's in 'overrides' now.
            # We need to process it from 'overrides' instead of kwargs if it exists.

            if "pop_ratio" in overrides and overrides["pop_ratio"] is not None:
                pop_ratio_val = overrides["pop_ratio"]
                # It might be string from click or float if default?
                # Click option type is str.
                pop_ratio_str = str(pop_ratio_val)
                if "," in pop_ratio_str:
                    pop_ratios = []
                    for pr_val in pop_ratio_str.split(","):
                        try:
                            pop_ratios.append(float(pr_val.strip()))
                        except ValueError:
                            if pr_val.strip().lower() == "inf":
                                pop_ratios.append(float("inf"))
                            else:
                                raise click.BadParameter(f"Invalid pop_ratio value: {pr_val}")
                    overrides["pop_ratio"] = pop_ratios
                else:
                    try:
                        overrides["pop_ratio"] = float(pop_ratio_str.strip())
                    except ValueError:
                        if pop_ratio_str.strip().lower() == "inf":
                            overrides["pop_ratio"] = float("inf")
                        else:
                            raise click.BadParameter(f"Invalid pop_ratio value: {pop_ratio_str}")

            eval_opts = base_eval_options.model_copy(update=overrides)
            kwargs["eval_options"] = eval_opts
            return func(*args, **kwargs)

        return wrapper

    if func is None:
        return decorator
    return decorator(func)


def heuristic_options(func: callable) -> callable:
    @click.option("-nn", "--neural_heuristic", is_flag=True, help="Use neural heuristic")
    @click.option(
        "--param-path",
        type=str,
        default=None,
        help="Path to the heuristic parameter file.",
    )
    @click.option(
        "--model-type",
        type=str,
        default=None,
        help="Type of the heuristic model.",
    )
    @click.option(
        "-q",
        "--use-quantize",
        is_flag=True,
        default=False,
        help="Use quantization (defaults to int8).",
    )
    @click.option(
        "--quant-type",
        type=click.Choice(["int8", "int4", "int4_w8a", "int8_w_only"]),
        default="int8",
        help="Specific AQT quantization configuration to use.",
    )
    @wraps(func)
    def wrapper(*args, **kwargs):
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
                raise click.UsageError(
                    f"Neural heuristic not available for puzzle '{kwargs['puzzle_name']}'."
                )

            model_type = heuristic_opts.model_type or "default"
            heuristic_config = heuristic_configs.get(model_type)
            if heuristic_config is None:
                raise click.UsageError(f"Neural heuristic config '{model_type}' not available.")

            param_path = heuristic_opts.param_path
            if param_path is None:
                path_template = heuristic_config.param_path
                if path_template is None:
                    raise click.UsageError(f"Parameter path for type '{model_type}' not found.")

                if "{size}" in path_template:
                    param_path = path_template.format(size=puzzle.size)
                else:
                    param_path = path_template

            heuristic: Heuristic = heuristic_config.callable(
                puzzle=puzzle,
                path=param_path,
                init_params=False,
                aqt_cfg=aqt_cfg,
            )
            attach_runtime_metadata(
                heuristic,
                model_type=model_type,
                param_path=param_path,
                extra={"aqt_cfg": aqt_cfg},
            )
        else:
            heuristic_callable = puzzle_bundle.heuristic
            if heuristic_callable is None:
                raise click.UsageError(
                    f"Heuristic not available for puzzle '{kwargs['puzzle_name']}'."
                )
            heuristic: Heuristic = heuristic_callable(puzzle)

        kwargs["heuristic"] = heuristic
        kwargs["heuristic_options"] = heuristic_opts
        return func(*args, **kwargs)

    return wrapper


def qfunction_options(func: callable) -> callable:
    @click.option("-nn", "--neural_qfunction", is_flag=True, help="Use neural q function")
    @click.option(
        "--param-path",
        type=str,
        default=None,
        help="Path to the Q-function parameter file.",
    )
    @click.option(
        "--model-type",
        type=str,
        default=None,
        help="Type of the Q-function model.",
    )
    @click.option(
        "-q",
        "--use-quantize",
        is_flag=True,
        default=False,
        help="Use quantization (defaults to int8).",
    )
    @click.option(
        "--quant-type",
        type=click.Choice(["int8", "int4", "int4_w8a", "int8_w_only"]),
        default="int8",
        help="Specific AQT quantization configuration to use.",
    )
    @wraps(func)
    def wrapper(*args, **kwargs):
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
                raise click.UsageError(
                    f"Neural Q-function not available for puzzle '{kwargs['puzzle_name']}'."
                )

            model_type = q_opts.model_type or "default"
            q_config = q_configs.get(model_type)
            if q_config is None:
                raise click.UsageError(f"Neural Q-function config '{model_type}' not available.")

            param_path = q_opts.param_path
            if param_path is None:
                path_template = q_config.param_path
                if path_template is None:
                    raise click.UsageError(f"Parameter path for type '{model_type}' not found.")

                if "{size}" in path_template:
                    param_path = path_template.format(size=puzzle.size)
                else:
                    param_path = path_template

            qfunction: QFunction = q_config.callable(
                puzzle=puzzle,
                path=param_path,
                init_params=False,
                aqt_cfg=aqt_cfg,
            )
            attach_runtime_metadata(
                qfunction,
                model_type=model_type,
                param_path=param_path,
                extra={"aqt_cfg": aqt_cfg},
            )
        else:
            q_callable = puzzle_bundle.q_function
            if q_callable is None:
                raise click.UsageError(
                    f"Q-function not available for puzzle '{kwargs['puzzle_name']}'."
                )
            qfunction: QFunction = q_callable(puzzle)

        kwargs["qfunction"] = qfunction
        kwargs["q_options"] = q_opts
        return func(*args, **kwargs)

    return wrapper


def visualize_options(func: callable) -> callable:
    @click.option(
        "-vt", "--visualize_terminal", is_flag=True, help="Visualize the path with terminal"
    )
    @click.option(
        "-vi", "--visualize_imgs", is_flag=True, help="Visualize the path with gif images"
    )
    @click.option("-mt", "--max_animation_time", default=10, type=int, help="Max animation time")
    @wraps(func)
    def wrapper(*args, **kwargs):
        vis_kwargs = map_kwargs_to_pydantic(VisualizeOptions, kwargs)
        vis_opts = VisualizeOptions(**vis_kwargs)
        kwargs["visualize_options"] = vis_opts
        return func(*args, **kwargs)

    return wrapper


def human_play_options(func: callable) -> callable:
    @wraps(func)
    def wrapper(*args, **kwargs):
        kwargs.pop("puzzle_bundle")
        kwargs.pop("puzzle_name")
        seeds = kwargs.pop("seeds")
        if len(seeds) > 1:
            raise ValueError("human play is not supported multiple initial state")
        kwargs["seed"] = seeds[0]
        return func(*args, **kwargs)

    return wrapper


def dist_train_options(
    func: callable = None, *, preset_category: str, default_preset: str | None = None
) -> callable:
    preset_map = train_presets.get(preset_category)
    if not preset_map:
        raise RuntimeError(f"Unknown training preset category '{preset_category}'.")

    preset_choices = list(preset_map.keys())
    preset_default = default_preset or preset_choices[0]
    if preset_default not in preset_map:
        raise RuntimeError(
            f"Default preset '{preset_default}' is not registered for category '{preset_category}'."
        )

    def decorator(inner: callable) -> callable:
        @click.option("-s", "--steps", type=HUMAN_INT, default=None)
        @click.option("-db", "--dataset_batch_size", type=HUMAN_INT, default=None)
        @click.option("-dmb", "--dataset_minibatch_size", type=HUMAN_INT, default=None)
        @click.option(
            "--sampling-non-backtracking-steps",
            type=HUMAN_INT,
            default=None,
            help="Number of previous states to avoid revisiting during dataset sampling.",
        )
        @click.option("-tmb", "--train_minibatch_size", type=HUMAN_INT, default=None)
        @click.option("-k", "--key", type=int, default=None)
        @click.option("-r", "--reset", type=bool, default=None)
        @click.option("-lt", "--loss_threshold", type=HUMAN_FLOAT, default=None)
        @click.option("-ui", "--update_interval", type=HUMAN_INT, default=None)
        @click.option("-fui", "--force_update_interval", type=HUMAN_INT, default=None)
        @click.option("-su", "--use_soft_update", is_flag=True, default=None)
        @click.option(
            "-ddn",
            "--use_double_dqn",
            is_flag=True,
            default=None,
            help="Enable Double DQN target computation.",
        )
        @click.option("-her", "--using_hindsight_target", is_flag=True, default=None)
        @click.option("-ts", "--using_triangular_sampling", is_flag=True, default=None)
        @click.option(
            "-dd",
            "--use_diffusion_distance",
            is_flag=True,
            default=None,
            help="Enable diffusion distance features in dataset creation.",
        )
        @click.option(
            "-ddm",
            "--use_diffusion_distance_mixture",
            is_flag=True,
            default=None,
            help="Enable diffusion distance mixture features in dataset creation.",
        )
        @click.option(
            "--use_diffusion_distance_warmup",
            is_flag=True,
            default=None,
            help="Enable warmup schedule when using diffusion distance features.",
        )
        @click.option(
            "--diffusion_distance_warmup_steps",
            type=HUMAN_INT,
            default=None,
            help="Number of iterations to run before enabling diffusion distance features.",
        )
        @click.option(
            "-tp",
            "--temperature",
            type=HUMAN_FLOAT,
            default=None,
            help="Boltzmann temperature for action selection.",
        )
        @click.option("-d", "--debug", is_flag=True, default=None)
        @click.option("-md", "--multi_device", type=bool, default=None)
        @click.option("-ri", "--reset_interval", type=HUMAN_INT, default=None)
        @click.option("-osr", "--opt_state_reset", type=bool, default=None)
        @click.option("--tau", type=HUMAN_FLOAT, default=None)
        @click.option(
            "--optimizer",
            type=click.Choice(list(OPTIMIZERS.keys())),
            default="normuon",
            help="Optimizer to use",
        )
        @click.option("-lr", "--learning_rate", type=HUMAN_FLOAT, default=None)
        @click.option(
            "-wd",
            "--weight_decay_size",
            type=HUMAN_FLOAT,
            default=None,
            help="Weight decay size for regularization.",
        )
        @click.option(
            "--loss",
            type=click.Choice(
                [
                    "mse",
                    "huber",
                    "logcosh",
                    "asymmetric_huber",
                    "asymmetric_logcosh",
                ]
            ),
            default=None,
            help="Select training loss.",
        )
        @click.option(
            "--loss-args",
            "loss_args",
            type=str,
            default=None,
            help=(
                "JSON object of additional keyword arguments for the selected loss, "
                'e.g. \'{"huber_delta":0.2,"asymmetric_tau":0.1}\'.'
            ),
        )
        @click.option(
            "--eval-search-metric",
            type=click.Choice(["astar", "astar_d", "beam", "qstar", "qbeam"]),
            default=None,
            help=(
                "Search algorithm to use for evaluation during training "
                "(heuristic: astar/astar_d/beam, qfunction: qstar/qbeam)."
            ),
        )
        @click.option(
            "-km",
            "--k_max",
            type=HUMAN_INT,
            default=None,
            help="Override puzzle's default k_max (formerly shuffle_length).",
        )
        @click.option(
            "--logger",
            type=click.Choice(["aim", "tensorboard", "wandb", "none"]),
            default=None,
            help="Logger to use.",
        )
        @click.option(
            "-pre",
            "--preset",
            type=click.Choice(preset_choices),
            default=preset_default,
            help=f"Training configuration preset for {preset_category.replace('_', ' ')}.",
        )
        @wraps(inner)
        def wrapper(*args, **kwargs):
            puzzle_bundle = kwargs["puzzle_bundle"]

            user_kmax = kwargs.pop("k_max")
            final_kmax = user_kmax if user_kmax is not None else puzzle_bundle.k_max
            # Pass through as k_max to downstream commands
            kwargs["k_max"] = final_kmax

            preset_name = kwargs.pop("preset")
            preset = preset_map[preset_name]

            # Collect any user-provided options to override the preset
            # map_kwargs_to_pydantic handles popping
            overrides = map_kwargs_to_pydantic(DistTrainOptions, kwargs)

            # Cleanup None values remaining in kwargs that correspond to DistTrainOptions fields
            for key in list(kwargs.keys()):
                if key in DistTrainOptions.model_fields and kwargs[key] is None:
                    kwargs.pop(key)

            # Handle special case for loss_args if it's a string in overrides
            if "loss_args" in overrides and isinstance(overrides["loss_args"], str):
                overrides["loss_args"] = json.loads(overrides["loss_args"])

            # Create a final options object by applying overrides to the preset
            train_opts = preset.model_copy(update=overrides)

            if train_opts.debug:
                print("Disabling JIT")
                jax.config.update("jax_disable_jit", True)

            kwargs["train_options"] = train_opts
            return inner(*args, **kwargs)

        return wrapper

    if func is None:
        return decorator
    return decorator(func)


def dist_heuristic_options(func: callable) -> callable:
    @click.option(
        "--param-path",
        type=str,
        default=None,
        help="Path to the heuristic parameter file.",
    )
    @click.option(
        "-nc",
        "--neural_config",
        type=str,
        default=None,
        help="Neural configuration. Overrides the default configuration.",
    )
    @click.option(
        "--model-type",
        type=str,
        default=None,
        help="Type of the heuristic model.",
    )
    @click.option(
        "-q",
        "--use-quantize",
        is_flag=True,
        default=False,
        help="Use quantization (defaults to int8).",
    )
    @click.option(
        "--quant-type",
        type=click.Choice(["int8", "int4", "int4_w8a", "int8_w_only"]),
        default="int8",
        help="Specific AQT quantization configuration to use.",
    )
    @wraps(func)
    def wrapper(*args, **kwargs):
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
        return func(*args, **kwargs)

    return wrapper


def dist_qfunction_options(func: callable) -> callable:
    @click.option(
        "--param-path",
        type=str,
        default=None,
        help="Path to the Q-function parameter file.",
    )
    @click.option(
        "-nc",
        "--neural_config",
        type=str,
        default=None,
        help="Neural configuration. Overrides the default configuration.",
    )
    @click.option(
        "--model-type",
        type=str,
        default=None,
        help="Type of the Q-function model.",
    )
    @click.option(
        "-q",
        "--use-quantize",
        is_flag=True,
        default=False,
        help="Use quantization (defaults to int8).",
    )
    @click.option(
        "--quant-type",
        type=click.Choice(["int8", "int4", "int4_w8a", "int8_w_only"]),
        default="int8",
        help="Specific AQT quantization configuration to use.",
    )
    @wraps(func)
    def wrapper(*args, **kwargs):
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
        return func(*args, **kwargs)

    return wrapper


def wm_get_ds_options(func: callable) -> callable:
    @click.option(
        "-ds",
        "--dataset",
        default="rubikscube",
        type=click.Choice(list(world_model_bundles.keys())),
        help="Dataset to use",
    )
    @wraps(func)
    def wrapper(*args, **kwargs):
        get_ds_kwargs = map_kwargs_to_pydantic(WMGetDSOptions, kwargs)
        get_ds_opts = WMGetDSOptions(**get_ds_kwargs)
        dataset_name = get_ds_opts.dataset
        wm_bundle = world_model_bundles[dataset_name]
        dataset_path = wm_bundle.dataset_path

        datas = jax.numpy.load(dataset_path + "/images.npy")
        next_datas = jax.numpy.load(dataset_path + "/next_images.npy")
        actions = jax.numpy.load(dataset_path + "/actions.npy")
        kwargs["datas"] = datas
        kwargs["next_datas"] = next_datas
        kwargs["actions"] = actions

        eval_trajectory = jax.numpy.load(dataset_path + "/eval_traj_images.npy")
        eval_actions = jax.numpy.load(dataset_path + "/eval_actions.npy")
        kwargs["eval_trajectory"] = (eval_trajectory, eval_actions)
        kwargs["get_ds_options"] = get_ds_opts
        return func(*args, **kwargs)

    return wrapper


def wm_get_world_model_options(func: callable) -> callable:
    @click.option(
        "--world_model",
        default="rubikscube",
        type=click.Choice(list(world_model_bundles.keys())),
        help="World model to use",
    )
    @wraps(func)
    def wrapper(*args, **kwargs):
        wm_model_kwargs = map_kwargs_to_pydantic(WMGetModelOptions, kwargs)
        wm_model_opts = WMGetModelOptions(**wm_model_kwargs)
        world_model_name = wm_model_opts.world_model
        wm_bundle = world_model_bundles[world_model_name]
        world_model = wm_bundle.world_model(reset=True)
        kwargs["world_model"] = world_model
        kwargs["world_model_name"] = world_model_name
        kwargs["wm_model_options"] = wm_model_opts
        return func(*args, **kwargs)

    return wrapper


def wm_train_options(func: callable) -> callable:
    @click.option("--train_epochs", type=HUMAN_INT, default=2000, help="Number of training steps")
    @click.option("--mini_batch_size", type=HUMAN_INT, default=1000, help="Batch size")
    @click.option(
        "--optimizer",
        type=click.Choice(list(OPTIMIZERS.keys())),
        default="adam",
        help="Optimizer to use",
    )
    @wraps(func)
    def wrapper(*args, **kwargs):
        wm_train_kwargs = map_kwargs_to_pydantic(WMTrainOptions, kwargs)
        wm_train_opts = WMTrainOptions(**wm_train_kwargs)
        kwargs["wm_train_options"] = wm_train_opts
        return func(*args, **kwargs)

    return wrapper


def wm_dataset_options(func: callable) -> callable:
    @click.option("--dataset_size", type=HUMAN_INT, default=300000)
    @click.option("--dataset_minibatch_size", type=HUMAN_INT, default=30000)
    @click.option("--shuffle_length", type=HUMAN_INT, default=30)
    @click.option("--img_size", nargs=2, type=click.Tuple([int, int]), default=(32, 32))
    @click.option("--key", type=int, default=0)
    @wraps(func)
    def wrapper(*args, **kwargs):
        # This decorator passes puzzle_name from another decorator,
        # but the puzzle object itself is what we need now.
        # Overwriting the puzzle in kwargs with the one for dataset generation
        puzzle_name = kwargs["puzzle_name"]
        if puzzle_name not in world_model_bundles:
            raise click.UsageError(
                f"World model dataset generation is not defined for '{puzzle_name}'"
            )

        wm_bundle = world_model_bundles[puzzle_name]
        puzzle_callable = wm_bundle.puzzle_for_ds_gen
        kwargs["puzzle"] = puzzle_callable()

        wm_dataset_kwargs = map_kwargs_to_pydantic(WMDatasetOptions, kwargs)
        wm_dataset_opts = WMDatasetOptions(**wm_dataset_kwargs)
        kwargs["wm_dataset_options"] = wm_dataset_opts
        return func(*args, **kwargs)

    return wrapper
