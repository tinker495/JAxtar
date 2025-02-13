from functools import wraps

import click
import jax

from heuristic.heuristic_base import Heuristic
from puzzle_config import (
    default_puzzle_sizes,
    puzzle_dict,
    puzzle_dict_hard,
    puzzle_heuristic_dict,
    puzzle_heuristic_dict_nn,
    puzzle_q_dict,
    puzzle_q_dict_nn,
)
from qfunction.q_base import QFunction


def puzzle_options(func: callable) -> callable:
    @click.option(
        "-p",
        "--puzzle",
        default="n-puzzle",
        type=click.Choice(puzzle_dict.keys()),
        help="Puzzle to solve",
    )
    @click.option("-h", "--hard", default=False, is_flag=True, help="Use the hard puzzle")
    @click.option("-ps", "--puzzle_size", default="default", type=str, help="Size of the puzzle")
    @click.option("--seeds", default="32", type=str, help="Seed for the random puzzle")
    @wraps(func)
    def wrapper(*args, **kwargs):
        if kwargs["puzzle_size"] == "default":
            kwargs["puzzle_size"] = default_puzzle_sizes[kwargs["puzzle"]]
        else:
            kwargs["puzzle_size"] = int(kwargs["puzzle_size"])

        puzzle_name = kwargs["puzzle"]
        kwargs["puzzle_name"] = puzzle_name
        if kwargs["hard"]:
            kwargs["puzzle"] = puzzle_dict_hard[puzzle_name](size=kwargs["puzzle_size"])
        else:
            kwargs["puzzle"] = puzzle_dict[puzzle_name](size=kwargs["puzzle_size"])
        kwargs.pop("hard")

        if kwargs["seeds"].isdigit():
            kwargs["seeds"] = [int(kwargs["seeds"])]
        else:
            try:
                kwargs["seeds"] = [int(s) for s in kwargs["seeds"].split(",")]
            except ValueError:
                raise ValueError("Invalid seeds")
        return func(*args, **kwargs)

    return wrapper


def search_options(func: callable) -> callable:
    @click.option("-m", "--max_node_size", default=2e6, help="Size of the puzzle")
    @click.option("-b", "--batch_size", default=8192, help="Batch size for BGPQ")  # 1024 * 8 = 8192
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
        return func(*args, **kwargs)

    return wrapper


def heuristic_options(func: callable) -> callable:
    @click.option("-nn", "--neural_heuristic", is_flag=True, help="Use neural heuristic")
    @wraps(func)
    def wrapper(*args, **kwargs):
        puzzle_name = kwargs["puzzle_name"]
        neural_heuristic = kwargs["neural_heuristic"]
        puzzle = kwargs["puzzle"]
        puzzle_size = kwargs["puzzle_size"]
        if neural_heuristic:
            try:
                heuristic: Heuristic = puzzle_heuristic_dict_nn[puzzle_name](
                    puzzle_size, puzzle, False
                )
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
        puzzle_size = kwargs["puzzle_size"]
        if neural_qfunction:
            try:
                qfunction: QFunction = puzzle_q_dict_nn[puzzle_name](puzzle_size, puzzle, False)
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
