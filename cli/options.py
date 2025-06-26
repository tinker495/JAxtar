import json
from functools import wraps

import click
import jax
import jax.numpy as jnp

from config import (
    puzzle_dict,
    puzzle_dict_ds,
    puzzle_dict_hard,
    puzzle_heuristic_dict,
    puzzle_heuristic_dict_nn,
    puzzle_q_dict,
    puzzle_q_dict_nn,
    world_model_dict,
    world_model_ds_dict,
)
from helpers.formatting import human_format_to_float
from heuristic.heuristic_base import Heuristic
from heuristic.neuralheuristic.neuralheuristic_base import NeuralHeuristicBase
from qfunction.neuralq.neuralq_base import NeuralQFunctionBase
from qfunction.q_base import QFunction


def create_puzzle_options(
    puzzle_dict_map,
    default_puzzle: str,
    use_hard_flag=False,
    puzzle_ds_flag=False,
    use_seeds_flag=False,
):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            input_args = {}
            if kwargs["puzzle_args"]:
                input_args = json.loads(kwargs["puzzle_args"])
            puzzle_name = kwargs["puzzle"]
            puzzle_size = kwargs["puzzle_size"]
            if puzzle_size != "default":
                puzzle_size = int(puzzle_size)
                input_args["size"] = puzzle_size

            if use_hard_flag and kwargs.get("hard"):
                puzzle_instance = puzzle_dict_hard[puzzle_name](**input_args)
            else:
                puzzle_instance = puzzle_dict_map[puzzle_name](**input_args)

            kwargs["puzzle"] = puzzle_instance
            kwargs["puzzle_name"] = puzzle_name
            kwargs["puzzle_size"] = puzzle_size

            if use_hard_flag:
                kwargs.pop("hard", None)

            if use_seeds_flag:
                if kwargs["seeds"].isdigit():
                    kwargs["seeds"] = [int(kwargs["seeds"])]
                else:
                    try:
                        kwargs["seeds"] = [int(s) for s in kwargs["seeds"].split(",")]
                    except ValueError:
                        raise ValueError("Invalid seeds")

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

        choices = list(puzzle_dict_map.keys())
        if use_hard_flag:
            # To avoid duplicates and preserve order
            hard_keys = [k for k in puzzle_dict_hard.keys() if k not in choices]
            choices.extend(hard_keys)
        elif puzzle_ds_flag:
            choices = list(puzzle_dict_ds.keys())

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
    puzzle_dict, default_puzzle="n-puzzle", use_hard_flag=True, use_seeds_flag=True
)
dist_puzzle_options = create_puzzle_options(
    puzzle_dict, default_puzzle="rubikscube", use_hard_flag=True
)
wm_puzzle_ds_options = create_puzzle_options(
    puzzle_dict_ds, default_puzzle="rubikscube", puzzle_ds_flag=True
)


def search_options(func: callable) -> callable:
    @click.option(
        "-m", "--max_node_size", default="2e6", type=str, help="Size of the puzzle"
    )  # this is a float for input like 2e6
    @click.option("-b", "--batch_size", default=int(1e4), type=int, help="Batch size for BGPQ")
    @click.option("-w", "--cost_weight", default=1.0 - 1e-3, help="Weight for the A* search")
    @click.option("-vm", "--vmap_size", default=1, help="Size for the vmap")
    @click.option("--debug", is_flag=True, help="Debug mode")
    @click.option("--profile", is_flag=True, help="Profile mode")
    @click.option("--show_compile_time", is_flag=True, help="Show compile time")
    @wraps(func)
    def wrapper(*args, **kwargs):
        if kwargs["debug"]:
            # disable jit
            print("Disabling JIT")
            jax.config.update("jax_disable_jit", True)

            # scale down the sizes for debugging
            kwargs["max_node_size"] = 10000
            kwargs["batch_size"] = 100
        kwargs.pop("debug")
        kwargs["max_node_size"] = int(human_format_to_float(kwargs["max_node_size"]))
        return func(*args, **kwargs)

    return wrapper


def heuristic_options(func: callable) -> callable:
    @click.option("-nn", "--neural_heuristic", is_flag=True, help="Use neural heuristic")
    @wraps(func)
    def wrapper(*args, **kwargs):
        puzzle_name = kwargs["puzzle_name"]
        neural_heuristic = kwargs["neural_heuristic"]
        puzzle = kwargs["puzzle"]
        if neural_heuristic:
            try:
                heuristic: Heuristic = puzzle_heuristic_dict_nn[puzzle_name](puzzle, False)
            except KeyError:
                print("Neural heuristic not available for this puzzle")
                print(f"list of neural heuristic: {puzzle_heuristic_dict_nn.keys()}")
                exit(1)
        else:
            heuristic: Heuristic = puzzle_heuristic_dict[puzzle_name](puzzle)
        kwargs["heuristic"] = heuristic
        kwargs.pop("neural_heuristic")
        kwargs.pop("puzzle_size")
        return func(*args, **kwargs)

    return wrapper


def qfunction_options(func: callable) -> callable:
    @click.option("-nn", "--neural_qfunction", is_flag=True, help="Use neural q function")
    @wraps(func)
    def wrapper(*args, **kwargs):
        puzzle_name = kwargs["puzzle_name"]
        neural_qfunction = kwargs["neural_qfunction"]
        puzzle = kwargs["puzzle"]
        if neural_qfunction:
            try:
                qfunction: QFunction = puzzle_q_dict_nn[puzzle_name](puzzle, False)
            except KeyError:
                print("Neural qfunction not available for this puzzle")
                print(f"list of neural qfunction: {puzzle_q_dict_nn.keys()}")
                exit(1)
        else:
            qfunction: QFunction = puzzle_q_dict[puzzle_name](puzzle)
        kwargs["qfunction"] = qfunction
        kwargs.pop("neural_qfunction")
        kwargs.pop("puzzle_size")
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
        return func(*args, **kwargs)

    return wrapper


def human_play_options(func: callable) -> callable:
    @wraps(func)
    def wrapper(*args, **kwargs):
        kwargs.pop("puzzle_name")
        kwargs.pop("puzzle_size")
        if len(kwargs["seeds"]) > 1:
            raise ValueError("human play is not supported multiple initial state")
        kwargs["seed"] = kwargs["seeds"][0]
        kwargs.pop("seeds")
        return func(*args, **kwargs)

    return wrapper


def dist_train_options(func: callable) -> callable:
    @click.option(
        "-s", "--steps", type=int, default=int(2e4)
    )  # 50 * 2e4 = 1e6 / DeepCubeA settings
    @click.option("-sl", "--shuffle_length", type=int, default=30)
    @click.option("-b", "--dataset_batch_size", type=int, default=524288)  # 8192 * 64
    @click.option("-mb", "--dataset_minibatch_size", type=int, default=8192)  # 128 * 16
    @click.option("-tmb", "--train_minibatch_size", type=int, default=8192)  # 128 * 16
    @click.option("-k", "--key", type=int, default=0)
    @click.option("-r", "--reset", is_flag=True, help="Reset the target heuristic params")
    @click.option("-l", "--loss_threshold", type=float, default=0.05)
    @click.option("-u", "--update_interval", type=int, default=128)
    @click.option("-su", "--use_soft_update", is_flag=True, help="Use soft update")
    @click.option("-her", "--using_hindsight_target", is_flag=True, help="Use hindsight target")
    @click.option(
        "-per", "--using_importance_sampling", is_flag=True, help="Use importance sampling"
    )
    @click.option("--debug", is_flag=True, help="Debug mode")
    @click.option("-m", "--multi_device", is_flag=True, help="Use multi device")
    @click.option("--reset_interval", type=int, default=4000, help="Reset interval")
    @click.option("--tau", type=float, default=0.2, help="Tau for scaled by reset")
    @wraps(func)
    def wrapper(*args, **kwargs):
        if kwargs["debug"]:
            # disable jit
            print("Disabling JIT")
            jax.config.update("jax_disable_jit", True)
        return func(*args, **kwargs)

    return wrapper


def dist_heuristic_options(func: callable) -> callable:
    @wraps(func)
    def wrapper(*args, **kwargs):
        puzzle_name = kwargs["puzzle_name"]
        puzzle_size = kwargs["puzzle_size"]
        puzzle = kwargs["puzzle"]
        reset = kwargs["reset"]
        try:
            heuristic: NeuralHeuristicBase = puzzle_heuristic_dict_nn[puzzle_name](
                puzzle_size, puzzle, reset
            )
        except KeyError:
            raise ValueError(f"No Neural Heuristic for {puzzle_name} with size {puzzle_size}")
        kwargs["heuristic"] = heuristic
        kwargs.pop("reset")
        return func(*args, **kwargs)

    return wrapper


def dist_qfunction_options(func: callable) -> callable:
    @click.option("-nwp", "--not_with_policy", is_flag=True, help="Not use policy for training")
    @wraps(func)
    def wrapper(*args, **kwargs):
        puzzle_name = kwargs["puzzle_name"]
        puzzle_size = kwargs["puzzle_size"]
        puzzle = kwargs["puzzle"]
        reset = kwargs["reset"]
        try:
            qfunction: NeuralQFunctionBase = puzzle_q_dict_nn[puzzle_name](
                puzzle_size, puzzle, reset
            )
        except KeyError:
            raise ValueError(f"No Neural Q Function for {puzzle_name} with size {puzzle_size}")
        kwargs["qfunction"] = qfunction
        kwargs["with_policy"] = not kwargs["not_with_policy"]
        kwargs.pop("not_with_policy")
        kwargs.pop("reset")
        return func(*args, **kwargs)

    return wrapper


def wm_get_ds_options(func: callable) -> callable:
    @click.option(
        "-ds",
        "--dataset",
        default="rubikscube",
        type=click.Choice(world_model_ds_dict.keys()),
        help="Dataset to use",
    )
    @wraps(func)
    def wrapper(*args, **kwargs):
        dataset_name = kwargs["dataset"]
        dataset_path = world_model_ds_dict[dataset_name]

        datas = jnp.load(dataset_path + "/images.npy")
        next_datas = jnp.load(dataset_path + "/next_images.npy")
        actions = jnp.load(dataset_path + "/actions.npy")
        kwargs["datas"] = datas
        kwargs["next_datas"] = next_datas
        kwargs["actions"] = actions

        eval_trajectory = jnp.load(dataset_path + "/eval_traj_images.npy")
        eval_actions = jnp.load(dataset_path + "/eval_actions.npy")
        kwargs["eval_trajectory"] = (eval_trajectory, eval_actions)
        return func(*args, **kwargs)

    return wrapper


def wm_get_world_model_options(func: callable) -> callable:
    @click.option(
        "--world_model",
        default="rubikscube",
        type=click.Choice(world_model_dict.keys()),
        help="World model to use",
    )
    @wraps(func)
    def wrapper(*args, **kwargs):
        world_model_name = kwargs["world_model"]
        world_model = world_model_dict[world_model_name]
        kwargs["world_model"] = world_model(reset=True)
        kwargs["world_model_name"] = world_model_name
        return func(*args, **kwargs)

    return wrapper


def wm_train_options(func: callable) -> callable:
    @click.option("--train_epochs", type=int, default=2000, help="Number of training steps")
    @click.option("--mini_batch_size", type=int, default=1000, help="Batch size")
    @wraps(func)
    def wrapper(*args, **kwargs):
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
        return func(*args, **kwargs)

    return wrapper
