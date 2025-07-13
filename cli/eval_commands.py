import json
import time
from collections.abc import MutableMapping
from datetime import datetime
from pathlib import Path

import click
import jax
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from puxle import Puzzle
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from xtructure import xtructure_numpy as xnp

from config.pydantic_models import EvalOptions, PuzzleOptions
from helpers import human_format
from helpers.config_printer import print_config
from helpers.plots import (
    plot_comparison_analysis,
    plot_heuristic_accuracy,
    plot_nodes_generated_by_path_cost,
    plot_path_cost_distribution,
    plot_pop_ratio_analysis,
    plot_search_time_by_path_cost,
)
from helpers.results import save_evaluation_results
from helpers.rich_progress import trange
from helpers.summaries import (
    create_comparison_summary_panel,
    create_pop_ratio_summary_panel,
    create_summary_panel,
)
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


def make_hashable(val):
    if isinstance(val, list):
        return tuple(val)
    if isinstance(val, dict):
        return json.dumps(val, sort_keys=True)
    return val


def display_value(val):
    # Convert tuples back to lists for display, and pretty-print JSON strings
    if isinstance(val, tuple):
        return str(list(val))
    try:
        loaded = json.loads(val)
        if isinstance(loaded, dict) or isinstance(loaded, list):
            return json.dumps(loaded, indent=2)
    except Exception:
        pass
    return str(val)


@click.group(name="eval")
def evaluation():
    """Evaluation commands"""
    pass


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

            states = []
            actual_dists = []
            estimated_dists = []
            for state_in_path in path:
                states.append(search_result.get_state(state_in_path))
                actual_dist = float(path_cost - search_result.get_cost(state_in_path))
                estimated_dist = float(search_result.get_dist(state_in_path))

                if np.isfinite(estimated_dist):
                    actual_dists.append(actual_dist)
                    estimated_dists.append(estimated_dist)

            result_item["path_analysis"] = {
                "actual": actual_dists,
                "estimated": estimated_dists,
                "states": xnp.concatenate(states),
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
    heuristic: Heuristic,
    eval_options: EvalOptions,
    puzzle_opts: PuzzleOptions,
    **kwargs,
):
    console = Console()
    heuristic_metadata = getattr(heuristic, "metadata", {})
    pop_ratios = (
        eval_options.pop_ratio
        if isinstance(eval_options.pop_ratio, list)
        else [eval_options.pop_ratio]
    )

    config = {
        "puzzle_options": puzzle_opts,
        "heuristic": heuristic.__class__.__name__,
        "heuristic_metadata": heuristic_metadata,
        "eval_options": eval_options,
        "num_eval": eval_options.num_eval,
        "pop_ratio_actual": pop_ratios,
    }
    print_config("Heuristic Evaluation Configuration", config)

    all_results = []
    base_run_name = (
        eval_options.run_name
        if eval_options.run_name
        else f"{puzzle_name}_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"
    )
    run_dir = Path("runs") / base_run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    for pr in pop_ratios:
        astar_fn = astar_builder(
            puzzle,
            heuristic,
            eval_options.batch_size,
            eval_options.get_max_node_size(),
            pop_ratio=pr,
            cost_weight=eval_options.cost_weight,
        )
        eval_seeds = list(range(eval_options.num_eval))
        results = run_evaluation(
            search_fn=astar_fn,
            puzzle=puzzle,
            seeds=eval_seeds,
        )
        for r in results:
            r["pop_ratio"] = pr
        all_results.extend(results)

    # Save results, print summary, and generate plots
    save_evaluation_results(
        results=all_results,
        run_dir=run_dir,
        config=config,
        save_per_pop_ratio=len(pop_ratios) > 1,
    )

    # Print config again after evaluation is complete
    print_config("Heuristic Evaluation Configuration", config)

    # Create a DataFrame for plotting
    solved_df = pd.DataFrame(all_results)
    solved_df = solved_df[solved_df["solved"]].copy()

    if len(pop_ratios) > 1:
        # Pop ratio summary and plots
        console.print(create_pop_ratio_summary_panel(all_results))
        if not solved_df.empty:
            plots = plot_pop_ratio_analysis(
                solved_df, scatter_max_points=eval_options.scatter_max_points
            )
            for plot_name, fig in plots.items():
                fig.savefig(run_dir / f"{plot_name}.png")
                plt.close(fig)
    else:
        # Single run summary and plots
        console.print(create_summary_panel(all_results))
        if not solved_df.empty:
            fig = plot_path_cost_distribution(solved_df)
            fig.savefig(run_dir / "path_cost_distribution.png")
            plt.close(fig)

            fig = plot_search_time_by_path_cost(solved_df)
            fig.savefig(run_dir / "search_time_by_path_cost.png")
            plt.close(fig)

            fig = plot_nodes_generated_by_path_cost(solved_df)
            fig.savefig(run_dir / "nodes_generated_by_path_cost.png")
            plt.close(fig)

    # Heuristic accuracy plot (always generated)
    fig = plot_heuristic_accuracy(all_results)
    fig.savefig(run_dir / "heuristic_accuracy.png")
    plt.close(fig)


@evaluation.command(name="qlearning")
@eval_puzzle_options
@eval_options
@qfunction_options
def eval_qlearning(
    puzzle: Puzzle,
    puzzle_name: str,
    qfunction: QFunction,
    eval_options: EvalOptions,
    puzzle_opts: PuzzleOptions,
    **kwargs,
):
    console = Console()
    qfunction_metadata = getattr(qfunction, "metadata", {})
    pop_ratios = (
        eval_options.pop_ratio
        if isinstance(eval_options.pop_ratio, list)
        else [eval_options.pop_ratio]
    )

    config = {
        "puzzle_options": puzzle_opts,
        "q_function": qfunction.__class__.__name__,
        "qfunction_metadata": qfunction_metadata,
        "eval_options": eval_options,
        "num_eval": eval_options.num_eval,
        "pop_ratio_actual": pop_ratios,
    }
    print_config("Q-Learning Evaluation Configuration", config)

    all_results = []
    base_run_name = (
        eval_options.run_name
        if eval_options.run_name
        else f"{puzzle_name}_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"
    )
    run_dir = Path("runs") / base_run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    for pr in pop_ratios:
        qstar_fn = qstar_builder(
            puzzle,
            qfunction,
            eval_options.batch_size,
            eval_options.get_max_node_size(),
            pop_ratio=pr,
            cost_weight=eval_options.cost_weight,
        )
        eval_seeds = list(range(eval_options.num_eval))
        results = run_evaluation(
            search_fn=qstar_fn,
            puzzle=puzzle,
            seeds=eval_seeds,
        )
        for r in results:
            r["pop_ratio"] = pr
        all_results.extend(results)

    # Save results, print summary, and generate plots
    save_evaluation_results(
        results=all_results,
        run_dir=run_dir,
        config=config,
        save_per_pop_ratio=len(pop_ratios) > 1,
    )

    # Print config again after evaluation is complete
    print_config("Q-Learning Evaluation Configuration", config)

    # Create a DataFrame for plotting
    solved_df = pd.DataFrame(all_results)
    solved_df = solved_df[solved_df["solved"]].copy()

    if len(pop_ratios) > 1:
        # Pop ratio summary and plots
        console.print(create_pop_ratio_summary_panel(all_results))
        if not solved_df.empty:
            plots = plot_pop_ratio_analysis(
                solved_df, scatter_max_points=eval_options.scatter_max_points
            )
            for plot_name, fig in plots.items():
                fig.savefig(run_dir / f"{plot_name}.png")
                plt.close(fig)
    else:
        # Single run summary and plots
        console.print(create_summary_panel(all_results))
        if not solved_df.empty:
            fig = plot_path_cost_distribution(solved_df)
            fig.savefig(run_dir / "path_cost_distribution.png")
            plt.close(fig)

            fig = plot_search_time_by_path_cost(solved_df)
            fig.savefig(run_dir / "search_time_by_path_cost.png")
            plt.close(fig)

            fig = plot_nodes_generated_by_path_cost(solved_df)
            fig.savefig(run_dir / "nodes_generated_by_path_cost.png")
            plt.close(fig)

    # Heuristic accuracy plot (always generated)
    fig = plot_heuristic_accuracy(all_results)
    fig.savefig(run_dir / "heuristic_accuracy.png")
    plt.close(fig)


def flatten_dict(d: MutableMapping, parent_key: str = "", sep: str = "."):
    items = []
    for k, v in d.items():
        new_key = parent_key + sep + k if parent_key else k
        if isinstance(v, MutableMapping):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)


@evaluation.command(name="compare")
@click.argument("run_dirs", nargs=-1, type=click.Path(exists=True, file_okay=False))
@click.option(
    "--scatter-max-points",
    type=int,
    default=2000,
    help="Maximum number of points to display on scatter plots.",
)
def eval_compare(run_dirs: list[str], scatter_max_points: int):
    """Compare multiple evaluation runs, including pop_ratio breakdown if present."""
    console = Console()
    all_dfs = []
    all_configs = {}
    composite_labels = []
    label_to_run = {}
    label_to_pop = {}

    for run_dir_str in run_dirs:
        run_dir = Path(run_dir_str)
        run_name = run_dir.name

        results_path = run_dir / "results.csv"
        if not results_path.exists():
            console.print(
                f"[bold red]Warning: Cannot find results.csv in {run_dir_str}. Skipping.[/bold red]"
            )
            continue

        config_path = run_dir / "config.json"
        if not config_path.exists():
            console.print(
                f"[bold red]Warning: Cannot find config.json in {run_dir_str}. Skipping.[/bold red]"
            )
            continue

        df = pd.read_csv(results_path)
        with open(config_path, "r") as f:
            config = json.load(f)
        all_configs[run_name] = config

        # If pop_ratio column exists and has >1 unique value, treat each as a sub-run
        if "pop_ratio" in df.columns and df["pop_ratio"].nunique() > 1:
            for pr, group in df.groupby("pop_ratio"):
                label = f"{run_name} (pop_ratio={pr})"
                group = group.copy()
                group["run_label"] = label
                all_dfs.append(group)
                composite_labels.append(label)
                label_to_run[label] = run_name
                label_to_pop[label] = pr
        else:
            label = run_name
            df["run_label"] = label
            all_dfs.append(df)
            composite_labels.append(label)
            label_to_run[label] = run_name
            label_to_pop[label] = df["pop_ratio"].iloc[0] if "pop_ratio" in df.columns else None

    if not all_dfs:
        console.print("[bold red]No valid evaluation runs found to compare.[/bold red]")
        return

    # Config comparison
    flat_configs = {name: flatten_dict(cfg) for name, cfg in all_configs.items()}
    config_df = pd.DataFrame.from_dict(flat_configs, orient="index")
    differing_params = []
    if not config_df.empty:
        for col in sorted(config_df.columns):
            try:
                if config_df[col].apply(make_hashable).nunique() > 1:
                    differing_params.append(col)
            except Exception as e:
                console.print(f"Skipping column {col} due to error: {e}", style="dim red")

    if differing_params:
        config_table = Table(
            title="[bold cyan]Configuration Differences[/bold cyan]",
            show_header=True,
            header_style="bold magenta",
        )
        config_table.add_column("Parameter", style="dim", width=30)
        for run_name in sorted(all_configs.keys()):
            config_table.add_column(run_name, justify="right")
        for param in differing_params:
            values = [
                display_value(config_df.loc[run_name, param])
                for run_name in sorted(all_configs.keys())
            ]
            config_table.add_row(param, *values)
        console.print(Panel(config_table, border_style="yellow", expand=False))
    else:
        console.print("[bold yellow]No configuration differences found among runs.[/bold yellow]")

    # Results comparison
    combined_df = pd.concat(all_dfs, ignore_index=True)

    run_labels = {}
    for label in composite_labels:
        run_name = label_to_run[label]
        pop_ratio = label_to_pop[label]
        diff_parts = []
        for param in differing_params:
            if "pop_ratio" in param:
                continue
            val = config_df.loc[run_name, param]
            diff_parts.append(f"{param.split('.')[-1]}={display_value(val)}")
        if pop_ratio is not None:
            diff_parts.append(f"pop_ratio={pop_ratio}")
        run_labels[label] = ", ".join(diff_parts) if diff_parts else "default"
    combined_df["run_label"] = combined_df["run_label"].map(run_labels)

    console.print(create_comparison_summary_panel(combined_df))

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    comparison_dir = Path("runs") / f"comparison_{timestamp}"
    comparison_dir.mkdir(parents=True, exist_ok=True)
    console.print(f"Saving comparison plots to [bold]{comparison_dir}[/bold]")

    solved_df = combined_df[combined_df["solved"]].copy()
    if not solved_df.empty:
        # Get a consistent order for labels in plots
        sorted_labels = sorted(combined_df["run_label"].unique())
        plots = plot_comparison_analysis(
            solved_df, sorted_labels, scatter_max_points=scatter_max_points
        )
        for plot_name, fig in plots.items():
            fig.savefig(comparison_dir / f"{plot_name}.png")
            plt.close(fig)
