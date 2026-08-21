import json
from functools import wraps

import click
import jax

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
from helpers.formatting import HUMAN_FLOAT, HUMAN_INT
from helpers.param_stats import attach_runtime_metadata
from helpers.util import map_kwargs_to_pydantic
from heuristic.heuristic_base import Heuristic
from qfunction.q_base import QFunction
from train_util.optimizer import OPTIMIZERS


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


def _build_puzzle(puzzle_opts: PuzzleOptions, *, default_hard: bool):
    puzzle_name = puzzle_opts.puzzle
    puzzle_bundle = _puzzle_bundles()[puzzle_name]

    input_args = json.loads(puzzle_opts.puzzle_args) if puzzle_opts.puzzle_args else {}
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
        raise click.UsageError(
            f"Puzzle type for '{puzzle_name}'"
            f"{' (hard)' if puzzle_opts.hard else ''} is not defined."
        )
    else:
        puzzle_instance = puzzle_callable(**input_args)

    return puzzle_name, puzzle_bundle, puzzle_instance


def create_puzzle_options(
    default_puzzle: str,
    default_hard=False,
    use_hard_flag=False,
    use_seeds_flag=False,
):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
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
            return func(*args, **kwargs)

        if use_seeds_flag:
            wrapper = click.option(
                "-s",
                "--seeds",
                default="0",
                type=str,
                help="Seed for the random puzzle",
            )(wrapper)

        wrapper = click.option(
            "-pargs",
            "--puzzle_args",
            default="",
            type=str,
            help="Arguments for the puzzle",
        )(wrapper)

        if use_hard_flag:
            wrapper = click.option(
                "-h", "--hard", default=False, is_flag=True, help="Use the hard puzzle"
            )(wrapper)

        wrapper = click.option(
            "-p",
            "--puzzle",
            default=default_puzzle,
            type=click.Choice(list(_puzzle_bundles().keys())),
            help="Puzzle to solve",
        )(wrapper)
        return wrapper

    return decorator


def benchmark_options(func: callable) -> callable:
    bundles = _benchmark_bundles()
    if not bundles:
        raise RuntimeError("No benchmark bundles registered.")

    default_benchmark = next(iter(bundles))

    @click.option(
        "--benchmark",
        "benchmark_key",
        default=None,
        type=click.Choice(list(bundles.keys())),
        help=f"Exact benchmark dataset. Defaults to '{default_benchmark}' unless --puzzle is set.",
    )
    @click.option(
        "-p",
        "--puzzle",
        "puzzle_key",
        default=None,
        type=click.Choice(list(_puzzle_bundles().keys())),
        help="Generate benchmark samples from a puzzle without an exact dataset.",
    )
    @click.option(
        "-pargs",
        "--puzzle-args",
        default="",
        type=str,
        help="Arguments for the generated puzzle.",
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
        puzzle_key = kwargs.pop("puzzle_key")
        puzzle_args = kwargs.pop("puzzle_args")
        benchmark_args_override = kwargs.pop("benchmark_args")
        sample_ids_raw = kwargs.pop("sample_ids")
        sample_limit = kwargs.pop("sample_limit")

        if benchmark_key and puzzle_key:
            raise click.UsageError("Use either --benchmark or --puzzle, not both.")
        if puzzle_args and not puzzle_key:
            raise click.UsageError("--puzzle-args requires --puzzle.")

        if puzzle_key:
            if benchmark_args_override or sample_ids_raw or sample_limit is not None:
                raise click.UsageError(
                    "--benchmark-args, --sample-ids, and --sample-limit require --benchmark."
                )
            puzzle_opts = PuzzleOptions(puzzle=puzzle_key, puzzle_args=puzzle_args)
            puzzle_name, puzzle_bundle, puzzle_instance = _build_puzzle(
                puzzle_opts, default_hard=True
            )
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
            return func(*args, **kwargs)

        benchmark_key = benchmark_key or default_benchmark
        benchmark_bundle = _benchmark_bundles()[benchmark_key]

        benchmark_args = dict(benchmark_bundle.benchmark_args or {})
        if benchmark_args_override:
            try:
                benchmark_args.update(json.loads(benchmark_args_override))
            except json.JSONDecodeError as exc:
                raise click.BadParameter(
                    f"Invalid JSON provided to --benchmark-args: {exc}"
                ) from exc

        benchmark_instance = benchmark_bundle.benchmark(**benchmark_args)

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
        kwargs["benchmark"] = benchmark_instance
        kwargs["benchmark_name"] = benchmark_key
        kwargs["benchmark_bundle"] = benchmark_bundle
        kwargs["benchmark_cli_options"] = {
            "sample_limit": sample_limit,
            "sample_ids": sample_ids,
        }
        kwargs["puzzle"] = benchmark_instance.puzzle
        kwargs["puzzle_name"] = benchmark_key
        kwargs["puzzle_bundle"] = benchmark_bundle
        kwargs["puzzle_opts"] = PuzzleOptions(puzzle=benchmark_key)
        return func(*args, **kwargs)

    return wrapper


puzzle_options = create_puzzle_options(
    default_puzzle="n-puzzle", use_hard_flag=True, use_seeds_flag=True
)


eval_puzzle_options = create_puzzle_options(default_puzzle="rubikscube", default_hard=True)

# dist training shares the eval puzzle option surface.
dist_puzzle_options = eval_puzzle_options


def search_options(func=None, *, variant: str = "default") -> callable:
    def decorator(func: callable) -> callable:
        @click.option(
            "-m",
            "--max_node_size",
            default=None,
            type=HUMAN_INT,
            help="Size of the puzzle",
        )
        @click.option(
            "-b",
            "--batch_size",
            default=None,
            type=HUMAN_INT,
            help="Batch size for BGPQ",
        )
        @click.option(
            "-w",
            "--cost_weight",
            default=None,
            type=HUMAN_FLOAT,
            help="Weight for the A* search",
        )
        @click.option(
            "-pr",
            "--pop_ratio",
            default=None,
            type=HUMAN_FLOAT,
            help="Ratio for popping nodes from the priority queue.",
        )
        @click.option(
            "--bound_step",
            default=None,
            type=HUMAN_FLOAT,
            help="ID{}* threshold grid size. 0 uses the exact next_bound ladder.",
        )
        @click.option(
            "--max_path_len",
            default=None,
            type=HUMAN_INT,
            help="ID{}* action-history length; also caps search depth. Dominates stack memory.",
        )
        @click.option("-vm", "--vmap_size", default=None, type=HUMAN_INT, help="Size for the vmap")
        @click.option("--debug", is_flag=True, default=None, help="Debug mode")
        @click.option("--profile", is_flag=True, default=None, help="Profile mode")
        @click.option("--show_compile_time", is_flag=True, default=None, help="Show compile time")
        @click.option(
            "--emit-workload-signature",
            "emit_workload_signature",
            is_flag=True,
            default=None,
            help="Emit Xtructure workload signature (xtr_* metrics).",
        )
        @click.option(
            "--search-preset",
            type=str,
            default=None,
            help="Name of the search preset to use.",
        )
        @wraps(func)
        def wrapper(*args, **kwargs):
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
                    raise click.UsageError(
                        f"Search preset '{search_preset}' not available for '{puzzle_name}'. "
                        f"Available: {list(search_options_configs.keys())}"
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

            return func(*args, **kwargs)

        return wrapper

    if func is None:
        return decorator
    return decorator(func)


def eval_options(func=None, *, variant: str = "default") -> callable:
    def decorator(func: callable) -> callable:
        @click.option(
            "-b",
            "--batch-size",
            type=HUMAN_INT,
            default=None,
            help="Batch size for search.",
        )
        @click.option("--show_compile_time", is_flag=True, default=None, help="Show compile time")
        @click.option(
            "--emit-workload-signature",
            "emit_workload_signature",
            is_flag=True,
            default=None,
            help="Emit Xtructure workload signature (xtr_* metrics).",
        )
        @click.option(
            "-m",
            "--max-node-size",
            type=HUMAN_INT,
            default=None,
            help="Maximum number of nodes to search.",
        )
        @click.option(
            "-w",
            "--cost-weight",
            type=HUMAN_FLOAT,
            default=None,
            help="Weight for cost in search.",
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
            "--bound_step",
            type=HUMAN_FLOAT,
            default=None,
            help="ID{}* threshold grid size. 0 uses the exact next_bound ladder.",
        )
        @click.option(
            "--max_path_len",
            default=None,
            type=HUMAN_INT,
            help="ID{}* action-history length; also caps search depth. Dominates stack memory.",
        )
        @click.option(
            "-ne",
            "--num-eval",
            type=HUMAN_INT,
            default=None,
            help="Number of puzzles to evaluate.",
        )
        @click.option(
            "-rn",
            "--run-name",
            type=str,
            default=None,
            help="Name of the evaluation run.",
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
        @click.option(
            "--eval-preset",
            type=str,
            default=None,
            help="Name of the evaluation preset to use.",
        )
        @wraps(func)
        def wrapper(*args, **kwargs):
            overrides = map_kwargs_to_pydantic(EvalOptions, kwargs)

            puzzle_bundle = kwargs["puzzle_bundle"]
            eval_preset = kwargs.pop("eval_preset")

            eval_options_configs = getattr(puzzle_bundle, "eval_options_configs", None)
            if eval_options_configs:
                preset_key = eval_preset or ("beam" if variant == "beam" else "default")
                if preset_key in eval_options_configs:
                    base_eval_options = eval_options_configs[preset_key]
                elif eval_preset:
                    puzzle_name = (
                        kwargs.get("puzzle_name") or kwargs.get("benchmark_name") or "unknown"
                    )
                    raise click.UsageError(
                        f"Eval preset '{eval_preset}' not available for '{puzzle_name}'. "
                        f"Available: {list(eval_options_configs.keys())}"
                    )
                else:
                    base_eval_options = EvalOptions()
            else:
                base_eval_options = EvalOptions()

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
                        except ValueError as e:
                            if pr_val.strip().lower() == "inf":
                                pop_ratios.append(float("inf"))
                            else:
                                raise click.BadParameter(
                                    f"Invalid pop_ratio value: {pr_val}"
                                ) from e
                    overrides["pop_ratio"] = pop_ratios
                else:
                    try:
                        overrides["pop_ratio"] = float(pop_ratio_str.strip())
                    except ValueError as e:
                        if pop_ratio_str.strip().lower() == "inf":
                            overrides["pop_ratio"] = float("inf")
                        else:
                            raise click.BadParameter(
                                f"Invalid pop_ratio value: {pop_ratio_str}"
                            ) from e

            eval_opts = base_eval_options.model_copy(update=overrides).resolve_for_eval_setup(
                has_benchmark=kwargs.get("benchmark") is not None
            )
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
        "-vt",
        "--visualize_terminal",
        is_flag=True,
        help="Visualize the path with terminal",
    )
    @click.option(
        "-vi",
        "--visualize_imgs",
        is_flag=True,
        help="Visualize the path with gif images",
    )
    @click.option("-mt", "--max_animation_time", default=10, type=int, help="Max animation time")
    @wraps(func)
    def wrapper(*args, **kwargs):
        vis_kwargs = map_kwargs_to_pydantic(VisualizeOptions, kwargs)
        vis_opts = VisualizeOptions(**vis_kwargs)
        kwargs["visualize_options"] = vis_opts
        return func(*args, **kwargs)

    return wrapper


def dist_train_options(
    func: callable = None, *, preset_category: str, default_preset: str | None = None
) -> callable:
    preset_map = _train_presets().get(preset_category)
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
            "--label",
            type=click.Choice(["td", "diffusion", "warmup_td"]),
            default=None,
            help=(
                "Training target generation: 'td' bootstrap targets (DAVI / Q-learning) "
                "min-capped by diffusion trajectory distances, 'diffusion' trajectory "
                "Bellman propagation, or 'warmup_td' diffusion targets for the first "
                "--warmup_ratio of steps before switching to td. Default: 'td'."
            ),
        )
        @click.option(
            "--warmup_ratio",
            type=HUMAN_FLOAT,
            default=None,
            help="Fraction of training steps using diffusion targets with --label warmup_td.",
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
            "-ec",
            "--eval_count",
            type=HUMAN_INT,
            default=None,
            help="Number of evaluations to perform during training (default: 5).",
        )
        @click.option(
            "--eval-search-metric",
            type=click.Choice(
                [
                    "astar",
                    "astar_d",
                    "bi_astar",
                    "bi_astar_d",
                    "beam",
                    "qstar",
                    "bi_qstar",
                    "qbeam",
                ]
            ),
            default=None,
            help=(
                "Search algorithm to use for evaluation during training "
                "(heuristic: astar/astar_d/bi_astar/bi_astar_d/beam, "
                "qfunction: qstar/bi_qstar/qbeam)."
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

            # Special handling for eval_options to support partial updates and defaults
            if "eval_options" in overrides:
                cli_eval = overrides["eval_options"]
                # If CLI passed an EvalOptions object (from @eval_options decorator)
                # and it's using the default num_eval (-1), revert to the preset's value.
                if hasattr(cli_eval, "num_eval") and cli_eval.num_eval == -1:
                    preset_eval_opts = getattr(preset, "eval_options", None)
                    if preset_eval_opts:
                        cli_eval.num_eval = preset_eval_opts.num_eval
                overrides["eval_options"] = cli_eval

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
