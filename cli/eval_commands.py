import time
from datetime import datetime
from pathlib import Path

import click
import jax
import jax.numpy as jnp
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
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

from .options import (
    eval_options,
    eval_puzzle_options,
    heuristic_options,
    qfunction_options,
)

matplotlib.use("Agg")


@click.group(name="eval")
def evaluation():
    """Evaluation commands"""
    pass


def log_and_summarize_evaluation(results: list[dict], run_dir: Path, console: Console):
    """Logs evaluation results to files and prints a summary."""
    run_dir.mkdir(parents=True, exist_ok=True)
    console.print(f"Saving results to [bold]{run_dir}[/bold]")

    df = pd.DataFrame(results)
    df.to_csv(run_dir / "results.csv", index=False)

    summary_table = Table(
        title="[bold cyan]Evaluation Summary[/bold cyan]",
        show_header=True,
        header_style="bold magenta",
    )
    summary_table.add_column("Metric", style="dim", width=30)
    summary_table.add_column("Value", justify="right")

    num_puzzles = len(results)
    solved_results = [r for r in results if r["solved"]]
    num_solved = len(solved_results)
    success_rate = (num_solved / num_puzzles) * 100 if num_puzzles > 0 else 0

    summary_table.add_row("Puzzles Attempted", str(num_puzzles))
    summary_table.add_row("Success Rate", f"{success_rate:.2f}% ({num_solved}/{num_puzzles})")

    if solved_results:
        solved_times = [r["search_time_s"] for r in solved_results]
        solved_nodes = [r["nodes_generated"] for r in solved_results]
        solved_paths = [r["path_length"] for r in solved_results]

        summary_table.add_row(
            "Avg. Search Time (Solved)", f"{jnp.mean(jnp.array(solved_times)):.3f} s"
        )
        summary_table.add_row(
            "Avg. Generated Nodes (Solved)", human_format(jnp.mean(jnp.array(solved_nodes)))
        )
        summary_table.add_row("Avg. Path Length", f"{jnp.mean(jnp.array(solved_paths)):.2f}")
    else:
        summary_table.add_row("Avg. Search Time (Solved)", "N/A")
        summary_table.add_row("Avg. Generated Nodes (Solved)", "N/A")
        summary_table.add_row("Avg. Path Length", "N/A")

    console.print(Panel(summary_table, border_style="green", expand=False))

    solved_df = df[df["solved"]]
    if not solved_df.empty:
        plt.figure(figsize=(10, 6))
        sns.histplot(data=solved_df, x="search_time_s", kde=True)
        plt.title("Distribution of Search Time")
        plt.xlabel("Search Time (s)")
        plt.ylabel("Frequency")
        plt.savefig(run_dir / "search_time_distribution.png")
        plt.close()

        plt.figure(figsize=(10, 6))
        sns.histplot(data=solved_df, x="nodes_generated", kde=True)
        plt.title("Distribution of Generated Nodes")
        plt.xlabel("Nodes Generated")
        plt.ylabel("Frequency")
        plt.savefig(run_dir / "nodes_generated_distribution.png")
        plt.close()

        plt.figure(figsize=(10, 6))
        sns.histplot(data=solved_df, x="path_length", kde=True)
        plt.title("Distribution of Path Length")
        plt.xlabel("Path Length")
        plt.ylabel("Frequency")
        plt.savefig(run_dir / "path_length_distribution.png")
        plt.close()

        fig, ax = plt.subplots(figsize=(10, 6))
        sns.boxplot(data=solved_df, x="path_length", y="search_time_s", ax=ax)
        sns.pointplot(
            data=solved_df,
            x="path_length",
            y="search_time_s",
            estimator=np.median,
            color="red",
            linestyles="--",
            errorbar=None,
            ax=ax,
        )
        ax.set_title("Search Time Distribution by Path Length")
        ax.set_xlabel("Path Length")
        ax.set_ylabel("Search Time (s)")
        plt.savefig(run_dir / "search_time_by_path_length.png")
        plt.close()

        fig, ax = plt.subplots(figsize=(10, 6))
        sns.boxplot(data=solved_df, x="path_length", y="nodes_generated", ax=ax)
        sns.pointplot(
            data=solved_df,
            x="path_length",
            y="nodes_generated",
            estimator=np.median,
            color="red",
            linestyles="--",
            errorbar=None,
            ax=ax,
        )
        ax.set_title("Generated Nodes Distribution by Path Length")
        ax.set_xlabel("Path Length")
        ax.set_ylabel("Nodes Generated")
        plt.savefig(run_dir / "nodes_generated_by_path_length.png")
        plt.close()


def run_evaluation(
    search_fn,
    puzzle: Puzzle,
    seeds: list[int],
) -> list[dict]:
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

        solved_results = [r for r in results if r["solved"]]
        num_solved = len(solved_results)
        success_rate = (num_solved / (i + 1)) * 100

        pbar_desc_dict = {"Success Rate": f"{success_rate:.2f}%"}
        if num_solved > 0:
            avg_time = sum(r["search_time_s"] for r in solved_results) / num_solved
            avg_nodes = sum(r["nodes_generated"] for r in solved_results) / num_solved
            pbar_desc_dict.update(
                {
                    "Avg Time (Solved)": f"{avg_time:.2f}s",
                    "Avg Nodes (Solved)": f"{human_format(avg_nodes)}",
                }
            )
        pbar.set_description("Evaluating", desc_dict=pbar_desc_dict)

    return results


@evaluation.command(name="heuristic")
@eval_puzzle_options
@eval_options
@heuristic_options
def eval_heuristic(
    puzzle: Puzzle,
    puzzle_name: str,
    eval_options: EvalOptions,
    heuristic: Heuristic,
    **kwargs,
):
    console = Console()
    config = {
        "puzzle": {"name": puzzle_name, "size": puzzle.size},
        "search_algorithm": "A*",
        "heuristic": heuristic.__class__.__name__,
        "eval_options": eval_options.dict(),
        "num_eval": eval_options.num_eval,
    }
    print_config("Heuristic Evaluation Configuration", config)

    astar_fn = astar_builder(
        puzzle,
        heuristic,
        eval_options.batch_size,
        eval_options.get_max_node_size(),
        cost_weight=eval_options.cost_weight,
    )

    eval_seeds = list(range(eval_options.num_eval))

    results = run_evaluation(
        search_fn=astar_fn,
        puzzle=puzzle,
        seeds=eval_seeds,
    )

    if eval_options.run_name:
        run_name = eval_options.run_name
    else:
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        run_name = f"{puzzle_name}_{timestamp}"

    run_dir = Path("runs") / run_name
    log_and_summarize_evaluation(results, run_dir, console)


@evaluation.command(name="qlearning")
@eval_puzzle_options
@eval_options
@qfunction_options
def eval_qlearning(
    puzzle: Puzzle,
    puzzle_name: str,
    eval_options: EvalOptions,
    qfunction: QFunction,
    **kwargs,
):
    console = Console()
    config = {
        "puzzle": {"name": puzzle_name, "size": puzzle.size},
        "search_algorithm": "Q*",
        "q_function": qfunction.__class__.__name__,
        "eval_options": eval_options.dict(),
        "num_eval": eval_options.num_eval,
    }
    print_config("Q-Learning Evaluation Configuration", config)

    qstar_fn = qstar_builder(
        puzzle,
        qfunction,
        eval_options.batch_size,
        eval_options.get_max_node_size(),
        cost_weight=eval_options.cost_weight,
    )

    eval_seeds = list(range(eval_options.num_eval))

    results = run_evaluation(
        search_fn=qstar_fn,
        puzzle=puzzle,
        seeds=eval_seeds,
    )

    if eval_options.run_name:
        run_name = eval_options.run_name
    else:
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        run_name = f"{puzzle_name}_{timestamp}"

    run_dir = Path("runs") / run_name
    log_and_summarize_evaluation(results, run_dir, console)
