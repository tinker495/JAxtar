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
    @click.option("--steps", type=int, default=int(1e6))
    @click.option("--shuffle_length", type=int, default=30)
    @click.option("--dataset_batch_size", type=int, default=int(1e6))
    @click.option("--dataset_minibatch_size", type=int, default=int(1e4))
    @click.option("--train_minibatch_size", type=int, default=int(1e3))
    @click.option("--key", type=int, default=0)
    @click.option("--reset", is_flag=True, help="Reset the target heuristic params")
    @click.option("-l", "--loss_threshold", type=float, default=float("inf"))
    @click.option("-u", "--update_interval", type=int, default=500)
    @click.option("--using_hindsight_target", is_flag=True, help="Use hindsight target")
    @click.option("--debug", is_flag=True, help="Debug mode")
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


def qfunction_options(func: callable) -> callable:
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
        kwargs.pop("reset")
        return func(*args, **kwargs)

    return wrapper
