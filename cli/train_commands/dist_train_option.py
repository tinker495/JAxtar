from functools import wraps

import click
import jax

from config import (
    default_puzzle_sizes,
    puzzle_dict,
    puzzle_dict_hard,
    puzzle_heuristic_dict_nn,
    puzzle_q_dict_nn,
)
from heuristic.neuralheuristic.neuralheuristic_base import NeuralHeuristicBase
from qfunction.neuralq.neuralq_base import NeuralQFunctionBase


def puzzle_options(func: callable) -> callable:
    @click.option(
        "-p",
        "--puzzle",
        default="rubikscube",
        type=click.Choice(puzzle_dict.keys()),
        help="Puzzle to solve",
    )
    @click.option("-h", "--hard", default=False, is_flag=True, help="Use the hard puzzle")
    @click.option("-ps", "--puzzle_size", default="default", type=str, help="Size of the puzzle")
    @wraps(func)
    def wrapper(*args, **kwargs):
        puzzle_name = kwargs["puzzle"]
        puzzle_size = kwargs["puzzle_size"]
        if puzzle_size == "default":
            puzzle_size = default_puzzle_sizes[puzzle_name]
        else:
            puzzle_size = int(puzzle_size)

        if kwargs["hard"]:
            kwargs["puzzle"] = puzzle_dict_hard[puzzle_name](size=puzzle_size)
        else:
            kwargs["puzzle"] = puzzle_dict[puzzle_name](size=puzzle_size)
        kwargs.pop("hard")
        kwargs["puzzle_name"] = puzzle_name
        kwargs["puzzle_size"] = puzzle_size
        return func(*args, **kwargs)

    return wrapper


def train_option(func: callable) -> callable:
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
    @wraps(func)
    def wrapper(*args, **kwargs):
        if kwargs["debug"]:
            # disable jit
            print("Disabling JIT")
            jax.config.update("jax_disable_jit", True)
        return func(*args, **kwargs)

    return wrapper


def heuristic_options(func: callable) -> callable:
    @wraps(func)
    def wrapper(*args, **kwargs):
        puzzle_name = kwargs["puzzle_name"]
        puzzle_size = kwargs["puzzle_size"]
        puzzle = kwargs["puzzle"]
        try:
            heuristic: NeuralHeuristicBase = puzzle_heuristic_dict_nn[puzzle_name](
                puzzle_size, puzzle, True
            )
        except KeyError:
            raise ValueError(f"No Neural Heuristic for {puzzle_name} with size {puzzle_size}")
        kwargs["heuristic"] = heuristic
        return func(*args, **kwargs)

    return wrapper


def qfunction_options(func: callable) -> callable:
    @wraps(func)
    def wrapper(*args, **kwargs):
        puzzle_name = kwargs["puzzle_name"]
        puzzle_size = kwargs["puzzle_size"]
        puzzle = kwargs["puzzle"]
        try:
            qfunction: NeuralQFunctionBase = puzzle_q_dict_nn[puzzle_name](
                puzzle_size, puzzle, True
            )
        except KeyError:
            raise ValueError(f"No Neural Q Function for {puzzle_name} with size {puzzle_size}")
        kwargs["qfunction"] = qfunction
        return func(*args, **kwargs)

    return wrapper


def train_wbs_option(func: callable) -> callable:
    @click.option(
        "-s", "--steps", type=int, default=int(2e3)
    )  # 50 * 2e4 = 1e6 / DeepCubeA settings
    @click.option("-sl", "--shuffle_length", type=int, default=30)
    @click.option("-rs", "--replay_size", type=int, default=int(2e7))
    @click.option("-ab", "--add_batch_size", type=int, default=524288)  # 8192 * 64
    @click.option("-sb", "--search_batch_size", type=int, default=1024)  # 8192 * 64
    @click.option("-tmb", "--train_minibatch_size", type=int, default=8192)  # 128 * 16
    @click.option("-w", "--cost_weight", type=float, default=0.8)
    @click.option("-k", "--key", type=int, default=0)
    @click.option("-r", "--reset", is_flag=True, help="Reset the target heuristic params")
    @click.option("-ob", "--use_optimal_branch", is_flag=True, help="Use optimal branch")
    @click.option("--debug", is_flag=True, help="Debug mode")
    @click.option("-m", "--multi_device", is_flag=True, help="Use multi device")
    @wraps(func)
    def wrapper(*args, **kwargs):
        if kwargs["debug"]:
            # disable jit
            print("Disabling JIT")
            jax.config.update("jax_disable_jit", True)
        return func(*args, **kwargs)

    return wrapper
