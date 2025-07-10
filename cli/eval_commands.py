import time

import click
import jax
import jax.numpy as jnp
from puxle import Puzzle
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from config.pydantic_models import EvalOptions
from helpers import human_format
from helpers.config_printer import print_config
from helpers.rich_progress import trange
from heuristic.heuristic_base import Heuristic
from JAxtar.astar import astar_builder
from JAxtar.qstar import qstar_builder
from qfunction.q_base import QFunction

from .options import eval_options, heuristic_options, puzzle_options, qfunction_options


@click.group(name="eval")
def evaluation():
    """Evaluation commands"""
    pass


def run_evaluation(
    search_fn,
    puzzle: Puzzle,
    seeds: list[int],
    eval_options: EvalOptions,
):
    console = Console()
    num_puzzles = len(seeds)
    results = []

    pbar = trange(
        num_puzzles,
        desc="Running Evaluations",
    )

    for i in pbar:
        seed = seeds[i]
        solve_config, state = puzzle.get_inits(jax.random.PRNGKey(seed))

        start_time = time.time()
        search_result = search_fn(solve_config, state)
        solved = bool(search_result.solved.block_until_ready())
        end_time = time.time()

        search_time = end_time - start_time
        generated_nodes = int(search_result.generated_size)

        result_item = {
            "seed": seed,
            "solved": solved,
            "search_time_s": search_time,
            "nodes_generated": generated_nodes,
            "path_length": 0,
        }

        if solved:
            path = search_result.get_solved_path()
            path_length = len(path)
            result_item["path_length"] = path_length

        results.append(result_item)

        # Update progress bar with moving averages
        num_solved = sum(r["solved"] for r in results)
        success_rate = (num_solved / (i + 1)) * 100
        avg_time = sum(r["search_time_s"] for r in results) / (i + 1)
        avg_nodes = sum(r["nodes_generated"] for r in results) / (i + 1)
        pbar.set_description(
            "Evaluating",
            desc_dict={
                "Success Rate": f"{success_rate:.2f}%",
                "Avg Time": f"{avg_time:.2f}s",
                "Avg Nodes": f"{human_format(avg_nodes)}",
            },
        )

    # Summary Table
    summary_table = Table(
        title="[bold cyan]Evaluation Summary[/bold cyan]",
        show_header=True,
        header_style="bold magenta",
    )
    summary_table.add_column("Metric", style="dim", width=25)
    summary_table.add_column("Value", justify="right")

    num_solved = sum(r["solved"] for r in results)
    success_rate = (num_solved / num_puzzles) * 100 if num_puzzles > 0 else 0
    total_times = [r["search_time_s"] for r in results]
    total_nodes = [r["nodes_generated"] for r in results]
    solved_paths = [r["path_length"] for r in results if r["solved"]]

    summary_table.add_row("Puzzles Attempted", str(num_puzzles))
    summary_table.add_row("Success Rate", f"{success_rate:.2f}% ({num_solved}/{num_puzzles})")
    summary_table.add_row("Avg. Search Time", f"{jnp.mean(jnp.array(total_times)):.3f} s")
    summary_table.add_row("Avg. Generated Nodes", human_format(jnp.mean(jnp.array(total_nodes))))
    if solved_paths:
        summary_table.add_row("Avg. Path Length", f"{jnp.mean(jnp.array(solved_paths)):.2f}")
    else:
        summary_table.add_row("Avg. Path Length", "N/A")

    console.print(Panel(summary_table, border_style="green", expand=False))

    return results


@evaluation.command(name="heuristic")
@eval_options
@puzzle_options
@heuristic_options
def eval_heuristic(
    puzzle: Puzzle,
    puzzle_name: str,
    seeds: tuple[int],
    eval_options: EvalOptions,
    heuristic: Heuristic,
    **kwargs,
):
    eval_seeds = (
        list(seeds) if len(seeds) > 1 else list(range(seeds[0], seeds[0] + eval_options.num_eval))
    )
    config = {
        "puzzle": {"name": puzzle_name, "size": puzzle.size},
        "search_algorithm": "A*",
        "heuristic": heuristic.__class__.__name__,
        "eval_options": eval_options.dict(),
        "num_eval": eval_options.num_eval,
        "seeds": tuple(eval_seeds),
    }
    print_config("Heuristic Evaluation Configuration", config)

    astar_fn = astar_builder(
        puzzle,
        heuristic,
        eval_options.batch_size,
        eval_options.get_max_node_size(),
        cost_weight=eval_options.cost_weight,
    )

    if not eval_seeds:
        print("No puzzles to evaluate. Please provide seeds or set --num-puzzles > 0.")
        return

    run_evaluation(
        search_fn=astar_fn,
        puzzle=puzzle,
        seeds=eval_seeds,
        eval_options=eval_options,
    )


@evaluation.command(name="qlearning")
@eval_options
@puzzle_options
@qfunction_options
def eval_qlearning(
    puzzle: Puzzle,
    puzzle_name: str,
    seeds: tuple[int],
    eval_options: EvalOptions,
    qfunction: QFunction,
    **kwargs,
):
    eval_seeds = (
        list(seeds) if len(seeds) > 1 else list(range(seeds[0], seeds[0] + eval_options.num_eval))
    )
    config = {
        "puzzle": {"name": puzzle_name, "size": puzzle.size},
        "search_algorithm": "Q*",
        "q_function": qfunction.__class__.__name__,
        "eval_options": eval_options.dict(),
        "num_eval": eval_options.num_eval,
        "seeds": tuple(eval_seeds),
    }
    print_config("Q-Learning Evaluation Configuration", config)

    qstar_fn = qstar_builder(
        puzzle,
        qfunction,
        eval_options.batch_size,
        eval_options.get_max_node_size(),
        cost_weight=eval_options.cost_weight,
    )

    if not eval_seeds:
        print("No puzzles to evaluate. Please provide seeds or set --num-puzzles > 0.")
        return

    run_evaluation(
        search_fn=qstar_fn,
        puzzle=puzzle,
        seeds=eval_seeds,
        eval_options=eval_options,
    )
