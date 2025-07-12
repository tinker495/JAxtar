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
        solved_paths = [r["path_cost"] for r in solved_results]

        summary_table.add_row(
            "Avg. Search Time (Solved)", f"{jnp.mean(jnp.array(solved_times)):.3f} s"
        )
        summary_table.add_row(
            "Avg. Generated Nodes (Solved)", human_format(jnp.mean(jnp.array(solved_nodes)))
        )
        summary_table.add_row("Avg. Path Cost", f"{jnp.mean(jnp.array(solved_paths)):.2f}")
    else:
        summary_table.add_row("Avg. Search Time (Solved)", "N/A")
        summary_table.add_row("Avg. Generated Nodes (Solved)", "N/A")
        summary_table.add_row("Avg. Path Cost", "N/A")

    console.print(Panel(summary_table, border_style="green", expand=False))

    solved_df = df[df["solved"]]
    if not solved_df.empty:
        plt.figure(figsize=(10, 6))
        sns.histplot(data=solved_df, x="path_cost", kde=True)
        plt.title("Distribution of Path Cost")
        plt.xlabel("Path Cost")
        plt.ylabel("Frequency")
        plt.savefig(run_dir / "path_cost_distribution.png")
        plt.close()

        fig, ax = plt.subplots(figsize=(10, 6))
        sns.boxplot(data=solved_df, x="path_cost", y="search_time_s", ax=ax)
        sns.pointplot(
            data=solved_df,
            x="path_cost",
            y="search_time_s",
            estimator=np.median,
            color="red",
            linestyles="--",
            errorbar=None,
            ax=ax,
        )
        ax.set_title("Search Time Distribution by Path Cost")
        ax.set_xlabel("Path Cost")
        ax.set_ylabel("Search Time (s)")
        plt.savefig(run_dir / "search_time_by_path_cost.png")
        plt.close()

        fig, ax = plt.subplots(figsize=(10, 6))
        sns.boxplot(data=solved_df, x="path_cost", y="nodes_generated", ax=ax)
        sns.pointplot(
            data=solved_df,
            x="path_cost",
            y="nodes_generated",
            estimator=np.median,
            color="red",
            linestyles="--",
            errorbar=None,
            ax=ax,
        )
        ax.set_title("Generated Nodes Distribution by Path Cost")
        ax.set_xlabel("Path Cost")
        ax.set_ylabel("Nodes Generated")
        plt.savefig(run_dir / "nodes_generated_by_path_cost.png")
        plt.close()

    all_actual_dists = []
    all_estimated_dists = []
    for r in solved_results:
        if r.get("path_analysis"):
            analysis_data = r["path_analysis"]
            if analysis_data.get("actual") and analysis_data.get("estimated"):
                all_actual_dists.extend(analysis_data["actual"])
                all_estimated_dists.extend(analysis_data["estimated"])

    if all_actual_dists:
        fig, ax = plt.subplots(figsize=(12, 12))
        plot_df = pd.DataFrame(
            {
                "Actual Cost to Goal": all_actual_dists,
                "Estimated Distance": all_estimated_dists,
            }
        )

        sns.boxplot(data=plot_df, x="Actual Cost to Goal", y="Estimated Distance", ax=ax)
        sns.pointplot(
            data=plot_df,
            x="Actual Cost to Goal",
            y="Estimated Distance",
            estimator=np.median,
            color="red",
            linestyles="--",
            errorbar=None,
            ax=ax,
        )

        max_val = 0
        if all_actual_dists and all_estimated_dists:
            max_val = max(np.max(all_actual_dists), np.max(all_estimated_dists))

        limit = int(max_val) + 1 if max_val > 0 else 10

        ax.plot(
            [0, limit], [0, limit], "g--", alpha=0.75, zorder=0, label="y=x (Perfect Heuristic)"
        )
        ax.set_xlim(0, limit)
        ax.set_ylim(0, limit)

        # Set x-axis ticks to cover the full range up to 'limit'
        xticks = range(int(limit) + 1)
        ax.set_xticks(xticks)
        ax.set_xticklabels([f"{x:.1f}" for x in xticks])
        # Reduce number of x-axis labels if there are too many
        xticklabels = ax.get_xticklabels()
        if len(xticklabels) > 10:
            step = int(np.ceil(len(xticklabels) / 10))
            for i, label in enumerate(xticklabels):
                if i % step != 0:
                    label.set_visible(False)

        ax.set_title("Heuristic/Q-function Accuracy Analysis")
        ax.set_xlabel("Actual Cost to Goal")
        ax.set_ylabel("Estimated Distance (Heuristic/Q-Value)")
        ax.legend()
        ax.grid(True)
        fig.tight_layout()
        fig.savefig(run_dir / "heuristic_accuracy.png")
        plt.close(fig)


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
            "path_cost": 0,
            "path_analysis": None,
        }

        if solved:
            path = search_result.get_solved_path()
            path_cost = search_result.get_cost(path[-1])
            result_item["path_cost"] = float(path_cost)

            actual_dists = []
            estimated_dists = []
            for state_in_path in path:
                actual_dist = float(path_cost - search_result.get_cost(state_in_path))
                estimated_dist = float(search_result.get_dist(state_in_path))

                if np.isfinite(estimated_dist):
                    actual_dists.append(actual_dist)
                    estimated_dists.append(estimated_dist)

            result_item["path_analysis"] = {
                "actual": actual_dists,
                "estimated": estimated_dists,
            }

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
        pop_ratio=eval_options.pop_ratio,
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
        pop_ratio=eval_options.pop_ratio,
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
