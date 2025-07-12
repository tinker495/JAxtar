import json
from functools import wraps

import click
import jax

from config import puzzle_bundles, train_presets, world_model_bundles
from config.pydantic_models import (
    DistQFunctionOptions,
    DistTrainOptions,
    EvalOptions,
    HeuristicOptions,
    PuzzleOptions,
    QFunctionOptions,
    SearchOptions,
    VisualizeOptions,
    WMDatasetOptions,
    WMGetDSOptions,
    WMGetModelOptions,
    WMTrainOptions,
)
from heuristic.heuristic_base import Heuristic
from heuristic.neuralheuristic.neuralheuristic_base import NeuralHeuristicBase
from neural_util.optimizer import OPTIMIZERS
from qfunction.neuralq.neuralq_base import NeuralQFunctionBase
from qfunction.q_base import QFunction


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
            puzzle_opt_keys = set(PuzzleOptions.model_fields.keys())
            present_keys = puzzle_opt_keys.intersection(kwargs.keys())
            puzzle_kwargs = {k: kwargs.pop(k) for k in present_keys}
            puzzle_opts = PuzzleOptions(**puzzle_kwargs)

            puzzle_name = puzzle_opts.puzzle
            puzzle_bundle = puzzle_bundles[puzzle_name]

            input_args = {}
            if puzzle_opts.puzzle_args:
                input_args = json.loads(puzzle_opts.puzzle_args)

            if puzzle_opts.puzzle_size != "default":
                input_args["size"] = int(puzzle_opts.puzzle_size)

            if (default_hard or puzzle_opts.hard) and puzzle_bundle.puzzle_hard is not None:
                puzzle_callable = puzzle_bundle.puzzle_hard
            elif puzzle_ds_flag:
                # This part is tricky. world_model_bundles has the specific puzzle_for_ds_gen
                # For now, let's assume the puzzle name exists in puzzle_bundles for ds-gen.
                # This will be handled more robustly in the world model options.
                puzzle_callable = puzzle_bundle.puzzle
            else:
                puzzle_callable = puzzle_bundle.puzzle

            if puzzle_callable is None:
                raise click.UsageError(
                    f"Puzzle type for '{puzzle_name}'"
                    f"{' (hard)' if puzzle_opts.hard else ''} is not defined."
                )

            puzzle_instance = puzzle_callable(**input_args)

            kwargs["puzzle"] = puzzle_instance
            kwargs["puzzle_name"] = puzzle_name
            kwargs["puzzle_bundle"] = puzzle_bundle

            if use_seeds_flag:
                kwargs["seeds"] = puzzle_opts.get_seed_list()

            return func(*args, **kwargs)

        if use_seeds_flag:
            wrapper = click.option(
                "-s", "--seeds", default="0", type=str, help="Seed for the random puzzle"
            )(wrapper)

        wrapper = click.option(
            "-pargs", "--puzzle_args", default="", type=str, help="Arguments for the puzzle"
        )(wrapper)
        wrapper = click.option(
            "-ps", "--puzzle_size", default="default", type=str, help="Size of the puzzle"
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


puzzle_options = create_puzzle_options(
    default_puzzle="n-puzzle", use_hard_flag=True, use_seeds_flag=True
)
eval_puzzle_options = create_puzzle_options(default_puzzle="n-puzzle", default_hard=True)
dist_puzzle_options = create_puzzle_options(default_puzzle="rubikscube", default_hard=True)
wm_puzzle_ds_options = create_puzzle_options(default_puzzle="rubikscube", puzzle_ds_flag=True)


def search_options(func: callable) -> callable:
    @click.option("-m", "--max_node_size", default=None, type=str, help="Size of the puzzle")
    @click.option("-b", "--batch_size", default=None, type=int, help="Batch size for BGPQ")
    @click.option("-w", "--cost_weight", default=None, type=float, help="Weight for the A* search")
    @click.option(
        "-pr",
        "--pop_ratio",
        default=None,
        type=float,
        help="Ratio for popping nodes from the priority queue.",
    )
    @click.option("-vm", "--vmap_size", default=None, type=int, help="Size for the vmap")
    @click.option("--debug", is_flag=True, default=None, help="Debug mode")
    @click.option("--profile", is_flag=True, default=None, help="Profile mode")
    @click.option("--show_compile_time", is_flag=True, default=None, help="Show compile time")
    @wraps(func)
    def wrapper(*args, **kwargs):
        puzzle_bundle = kwargs["puzzle_bundle"]
        base_search_options = puzzle_bundle.search_options

        overrides = {
            k: v for k, v in kwargs.items() if v is not None and k in SearchOptions.model_fields
        }
        search_opts = base_search_options.model_copy(update=overrides)

        if search_opts.debug:
            print("Disabling JIT")
            jax.config.update("jax_disable_jit", True)
            search_opts.max_node_size = 10000
            search_opts.batch_size = 100

        kwargs["search_options"] = search_opts

        for k in SearchOptions.model_fields:
            kwargs.pop(k, None)
        return func(*args, **kwargs)

    return wrapper


def eval_options(func: callable) -> callable:
    @click.option("--batch-size", type=int, default=None, help="Batch size for search.")
    @click.option(
        "--max-node-size", type=int, default=None, help="Maximum number of nodes to search."
    )
    @click.option("--cost-weight", type=float, default=None, help="Weight for cost in search.")
    @click.option(
        "-pr",
        "--pop_ratio",
        type=float,
        default=None,
        help="Ratio for popping nodes from the priority queue.",
    )
    @click.option("--num-eval", type=int, default=None, help="Number of puzzles to evaluate.")
    @click.option("--run-name", type=str, default=None, help="Name of the evaluation run.")
    @wraps(func)
    def wrapper(*args, **kwargs):
        puzzle_bundle = kwargs["puzzle_bundle"]
        base_eval_options = puzzle_bundle.eval_options
        # Collect any user-provided options to override the preset
        overrides = {
            k: v for k, v in kwargs.items() if v is not None and k in EvalOptions.model_fields
        }
        eval_opts = base_eval_options.model_copy(update=overrides)
        kwargs["eval_options"] = eval_opts
        for k in EvalOptions.model_fields:
            kwargs.pop(k, None)
        return func(*args, **kwargs)

    return wrapper


def heuristic_options(func: callable) -> callable:
    @click.option("-nn", "--neural_heuristic", is_flag=True, help="Use neural heuristic")
    @wraps(func)
    def wrapper(*args, **kwargs):
        heuristic_opts = HeuristicOptions(
            **{k: kwargs.pop(k) for k in HeuristicOptions.model_fields}
        )
        puzzle_bundle = kwargs.pop("puzzle_bundle")
        puzzle = kwargs["puzzle"]
        is_eval = kwargs.get("eval_options", None) is not None

        if heuristic_opts.neural_heuristic or is_eval:
            heuristic_callable = puzzle_bundle.heuristic_nn
            if heuristic_callable is None:
                raise click.UsageError(
                    f"Neural heuristic not available for puzzle '{kwargs['puzzle_name']}'."
                )
            heuristic: Heuristic = heuristic_callable(puzzle, False)
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
    @wraps(func)
    def wrapper(*args, **kwargs):
        q_opts = QFunctionOptions(**{k: kwargs.pop(k) for k in QFunctionOptions.model_fields})
        puzzle_bundle = kwargs.pop("puzzle_bundle")
        puzzle = kwargs["puzzle"]
        is_eval = kwargs.get("eval_options", None) is not None

        if q_opts.neural_qfunction or is_eval:
            q_callable = puzzle_bundle.q_function_nn
            if q_callable is None:
                raise click.UsageError(
                    f"Neural Q-function not available for puzzle '{kwargs['puzzle_name']}'."
                )
            qfunction: QFunction = q_callable(puzzle, False)
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
        vis_opts = VisualizeOptions(**{k: kwargs.pop(k) for k in VisualizeOptions.model_fields})
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


def dist_train_options(func: callable) -> callable:
    @click.option("-s", "--steps", type=int, default=None)
    @click.option("-db", "--dataset_batch_size", type=int, default=None)
    @click.option("-dmb", "--dataset_minibatch_size", type=int, default=None)
    @click.option("-tmb", "--train_minibatch_size", type=int, default=None)
    @click.option("-k", "--key", type=int, default=None)
    @click.option("-r", "--reset", type=bool, default=None)
    @click.option("-lt", "--loss_threshold", type=float, default=None)
    @click.option("-ui", "--update_interval", type=int, default=None)
    @click.option("-su", "--use_soft_update", is_flag=True, default=None)
    @click.option("-her", "--using_hindsight_target", is_flag=True, default=None)
    @click.option("-is", "--using_importance_sampling", is_flag=True, default=None)
    @click.option("-ts", "--using_triangular_sampling", is_flag=True, default=None)
    @click.option(
        "-tcw",
        "--target_confidence_weighting",
        "use_target_confidence_weighting",
        is_flag=True,
        default=None,
        help="Weight loss by target confidence (inverse of move_cost).",
    )
    @click.option("-d", "--debug", is_flag=True, default=None)
    @click.option("-md", "--multi_device", type=bool, default=None)
    @click.option("-ri", "--reset_interval", type=int, default=None)
    @click.option("-osr", "--opt_state_reset", type=bool, default=None)
    @click.option("-t", "--tau", type=float, default=None)
    @click.option(
        "--optimizer",
        type=click.Choice(list(OPTIMIZERS.keys())),
        default="adam",
        help="Optimizer to use",
    )
    @click.option(
        "-sl",
        "--shuffle_length",
        type=int,
        default=None,
        help="Override puzzle's default shuffle length.",
    )
    @click.option(
        "-pre",
        "--preset",
        type=click.Choice(list(train_presets.keys())),
        default="default",
        help="Training configuration preset.",
    )
    @wraps(func)
    def wrapper(*args, **kwargs):
        puzzle_bundle = kwargs["puzzle_bundle"]

        user_shuffle_length = kwargs.pop("shuffle_length")
        final_shuffle_length = (
            user_shuffle_length if user_shuffle_length is not None else puzzle_bundle.shuffle_length
        )
        kwargs["shuffle_length"] = final_shuffle_length

        preset_name = kwargs.pop("preset")
        preset = train_presets[preset_name]

        # Collect any user-provided options to override the preset
        overrides = {
            k: v for k, v in kwargs.items() if v is not None and k in DistTrainOptions.model_fields
        }

        # Create a final options object by applying overrides to the preset
        train_opts = preset.model_copy(update=overrides)

        # Clean up kwargs so they don't get passed down
        for k in DistTrainOptions.model_fields:
            kwargs.pop(k, None)

        if train_opts.debug:
            print("Disabling JIT")
            jax.config.update("jax_disable_jit", True)

        kwargs["train_options"] = train_opts
        return func(*args, **kwargs)

    return wrapper


def dist_heuristic_options(func: callable) -> callable:
    @wraps(func)
    def wrapper(*args, **kwargs):
        puzzle_bundle = kwargs["puzzle_bundle"]
        puzzle = kwargs["puzzle"]
        reset = kwargs["train_options"].reset

        heuristic_callable = puzzle_bundle.heuristic_nn
        if heuristic_callable is None:
            raise click.UsageError(
                f"Neural heuristic not available for puzzle '{kwargs['puzzle_name']}'."
            )

        heuristic: NeuralHeuristicBase = heuristic_callable(puzzle, reset)
        kwargs["heuristic"] = heuristic
        return func(*args, **kwargs)

    return wrapper


def dist_qfunction_options(func: callable) -> callable:
    @click.option("-nwp", "--not_with_policy", is_flag=True, help="Not use policy for training")
    @wraps(func)
    def wrapper(*args, **kwargs):
        q_opts = DistQFunctionOptions(
            **{k: kwargs.pop(k) for k in DistQFunctionOptions.model_fields}
        )
        puzzle_bundle = kwargs["puzzle_bundle"]
        puzzle = kwargs["puzzle"]
        reset = kwargs["train_options"].reset

        q_callable = puzzle_bundle.q_function_nn
        if q_callable is None:
            raise click.UsageError(
                f"Neural Q-function not available for puzzle '{kwargs['puzzle_name']}'."
            )

        qfunction: NeuralQFunctionBase = q_callable(puzzle, reset)
        kwargs["qfunction"] = qfunction
        kwargs["with_policy"] = not q_opts.not_with_policy
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
        get_ds_opts = WMGetDSOptions(**{k: kwargs.pop(k) for k in WMGetDSOptions.model_fields})
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
        wm_model_opts = WMGetModelOptions(
            **{k: kwargs.pop(k) for k in WMGetModelOptions.model_fields}
        )
        world_model_name = wm_model_opts.world_model
        wm_bundle = world_model_bundles[world_model_name]
        world_model = wm_bundle.world_model(reset=True)
        kwargs["world_model"] = world_model
        kwargs["world_model_name"] = world_model_name
        kwargs["wm_model_options"] = wm_model_opts
        return func(*args, **kwargs)

    return wrapper


def wm_train_options(func: callable) -> callable:
    @click.option("--train_epochs", type=int, default=2000, help="Number of training steps")
    @click.option("--mini_batch_size", type=int, default=1000, help="Batch size")
    @click.option(
        "--optimizer",
        type=click.Choice(list(OPTIMIZERS.keys())),
        default="adam",
        help="Optimizer to use",
    )
    @wraps(func)
    def wrapper(*args, **kwargs):
        wm_train_opts = WMTrainOptions(**{k: kwargs.pop(k) for k in WMTrainOptions.model_fields})
        kwargs["wm_train_options"] = wm_train_opts
        return func(*args, **kwargs)

    return wrapper


def wm_dataset_options(func: callable) -> callable:
    @click.option("--dataset_size", type=int, default=300000)
    @click.option("--dataset_minibatch_size", type=int, default=30000)
    @click.option("--shuffle_length", type=int, default=30)
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

        wm_dataset_opts = WMDatasetOptions(
            **{k: kwargs.pop(k) for k in WMDatasetOptions.model_fields}
        )
        kwargs["wm_dataset_options"] = wm_dataset_opts
        return func(*args, **kwargs)

    return wrapper
