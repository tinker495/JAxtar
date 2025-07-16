import itertools
import json
import time
from collections.abc import MutableMapping
from datetime import datetime
from pathlib import Path
from typing import Callable, List, Union

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
    plot_expansion_distribution,
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
            "expansion_analysis": None,
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

        # Extract expansion data for plotting node value distributions
        expanded_nodes_mask = search_result.pop_generation > -1
        # Use np.asarray to handle potential JAX arrays on different devices
        if np.any(np.asarray(expanded_nodes_mask)):
            pop_generations = np.asarray(search_result.pop_generation[expanded_nodes_mask])
            costs = np.asarray(search_result.cost[expanded_nodes_mask])
            dists = np.asarray(search_result.dist[expanded_nodes_mask])

            if pop_generations.size > 0:
                result_item["expansion_analysis"] = {
                    "pop_generation": pop_generations,
                    "cost": costs,
                    "dist": dists,
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


def _run_evaluation_sweep(
    puzzle: Puzzle,
    puzzle_name: str,
    search_model: Union[Heuristic, QFunction],
    search_model_name: str,
    search_builder_fn: Callable,
    eval_options: EvalOptions,
    puzzle_opts: PuzzleOptions,
    **kwargs,
):
    console = Console()
    model_metadata = getattr(search_model, "metadata", {})

    pop_ratios = (
        eval_options.pop_ratio
        if isinstance(eval_options.pop_ratio, list)
        else [eval_options.pop_ratio]
    )
    cost_weights = (
        eval_options.cost_weight
        if isinstance(eval_options.cost_weight, list)
        else [eval_options.cost_weight]
    )
    batch_sizes = (
        eval_options.batch_size
        if isinstance(eval_options.batch_size, list)
        else [eval_options.batch_size]
    )

    param_combinations = list(itertools.product(pop_ratios, cost_weights, batch_sizes))
    is_sweep = len(param_combinations) > 1

    base_run_name = (
        eval_options.run_name
        if eval_options.run_name
        else f"{puzzle_name}_{search_model_name}_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"
    )
    main_run_dir = Path("runs") / base_run_name
    main_run_dir.mkdir(parents=True, exist_ok=True)

    if is_sweep:
        console.print(
            f"Starting parameter sweep with {len(param_combinations)} combinations."
            f"Results will be in [bold]{main_run_dir}[/bold]"
        )

    sub_run_dirs = []
    for i, (pr, cw, bs) in enumerate(param_combinations):
        run_dir = main_run_dir
        if is_sweep:
            run_name = f"pr_{pr}_cw_{cw}_bs_{bs}".replace("inf", "Infinity")
            run_dir = main_run_dir / run_name
            sub_run_dirs.append(str(run_dir))
        run_dir.mkdir(parents=True, exist_ok=True)

        current_eval_opts = eval_options.copy(
            update={"pop_ratio": pr, "cost_weight": cw, "batch_size": bs}
        )

        config = {
            "puzzle_options": puzzle_opts.dict(),
            search_model_name: search_model.__class__.__name__,
            f"{search_model_name}_metadata": model_metadata,
            "eval_options": current_eval_opts.dict(),
        }

        if is_sweep:
            console.rule(
                f"[bold cyan]Run {i+1}/{len(param_combinations)}: pr={pr}, cw={cw}, bs={bs}[/bold cyan]"
            )
        print_config(f"{search_model_name.capitalize()} Evaluation Configuration", config)

        search_fn = search_builder_fn(
            puzzle,
            search_model,
            bs,
            eval_options.get_max_node_size(bs),
            pop_ratio=pr,
            cost_weight=cw,
        )

        eval_seeds = list(range(eval_options.num_eval))
        results = run_evaluation(
            search_fn=search_fn,
            puzzle=puzzle,
            seeds=eval_seeds,
        )
        for r in results:
            r["pop_ratio"] = pr
            r["cost_weight"] = cw
            r["batch_size"] = bs

        save_evaluation_results(
            results=results, run_dir=run_dir, config=config, save_per_pop_ratio=False
        )
        print_config(f"{search_model_name.capitalize()} Evaluation Configuration", config)

        solved_df = pd.DataFrame([r for r in results if r["solved"]])
        console.print(create_summary_panel(results))

        if not solved_df.empty:
            if len(pop_ratios) > 1 and not is_sweep:
                console.print(create_pop_ratio_summary_panel(results))
                plots = plot_pop_ratio_analysis(
                    solved_df, scatter_max_points=eval_options.scatter_max_points
                )
                for plot_name, fig in plots.items():
                    fig.savefig(run_dir / f"{plot_name}.png")
                    plt.close(fig)
            else:
                fig = plot_path_cost_distribution(solved_df)
                fig.savefig(run_dir / "path_cost_distribution.png")
                plt.close(fig)

                fig = plot_search_time_by_path_cost(solved_df)
                fig.savefig(run_dir / "search_time_by_path_cost.png")
                plt.close(fig)

                fig = plot_nodes_generated_by_path_cost(solved_df)
                fig.savefig(run_dir / "nodes_generated_by_path_cost.png")
                plt.close(fig)

        fig = plot_heuristic_accuracy(results)
        fig.savefig(run_dir / "heuristic_accuracy.png")
        plt.close(fig)

        expansion_plot_dir = run_dir / "expansion_plots"
        expansion_plot_dir.mkdir(exist_ok=True)
        plots_generated = 0
        for r in results:
            if plots_generated >= current_eval_opts.max_expansion_plots:
                break
            if r.get("expansion_analysis"):
                fig = plot_expansion_distribution(
                    [r], scatter_max_points=current_eval_opts.scatter_max_points
                )
                fig.savefig(expansion_plot_dir / f"expansion_dist_seed_{r['seed']}.png")
                plt.close(fig)
                plots_generated += 1

    if is_sweep:
        console.rule("[bold green]Sweep Complete. Generating Comparison Report.[/bold green]")
        _generate_comparison_report(
            run_dirs=sub_run_dirs,
            output_dir=main_run_dir,
            scatter_max_points=eval_options.scatter_max_points,
        )
        console.print(f"Comparison report saved in [bold]{main_run_dir}[/bold]")


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
    """Evaluate a heuristic with optional parameter sweeps."""
    _run_evaluation_sweep(
        puzzle=puzzle,
        puzzle_name=puzzle_name,
        search_model=heuristic,
        search_model_name="heuristic",
        search_builder_fn=astar_builder,
        eval_options=eval_options,
        puzzle_opts=puzzle_opts,
        **kwargs,
    )


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
    """Evaluate a Q-function with optional parameter sweeps."""
    _run_evaluation_sweep(
        puzzle=puzzle,
        puzzle_name=puzzle_name,
        search_model=qfunction,
        search_model_name="qfunction",
        search_builder_fn=qstar_builder,
        eval_options=eval_options,
        puzzle_opts=puzzle_opts,
        **kwargs,
    )


def flatten_dict(d: MutableMapping, parent_key: str = "", sep: str = "."):
    items = []
    for k, v in d.items():
        new_key = parent_key + sep + k if parent_key else k
        if isinstance(v, MutableMapping):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)


def _generate_comparison_report(run_dirs: List[str], output_dir: Path, scatter_max_points: int):
    console = Console()
    all_dfs = []
    all_configs = {}
    run_labels = {}

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

        df["run_label"] = run_name
        all_dfs.append(df)
        all_configs[run_name] = config

    if not all_dfs:
        console.print("[bold red]No valid evaluation runs found to compare.[/bold red]")
        return

    # Config comparison
    flat_configs = {name: flatten_dict(cfg) for name, cfg in all_configs.items()}
    config_df = pd.DataFrame.from_dict(flat_configs, orient="index")
    differing_params = []
    if not config_df.empty:
        # Prioritize key eval options for comparison
        priority_params = [
            "eval_options.pop_ratio",
            "eval_options.cost_weight",
            "eval_options.batch_size",
        ]
        other_params = sorted(
            [
                col
                for col in config_df.columns
                if col not in priority_params
                and "metadata" not in col
                and "puzzle_options" not in col
            ]
        )

        for col in priority_params + other_params:
            if col not in config_df.columns:
                continue
            try:
                if config_df[col].apply(make_hashable).nunique() > 1:
                    differing_params.append(col)
            except Exception as e:
                console.print(f"Skipping column {col} due to error: {e}", style="dim red")

    if differing_params:
        for run_name, row in config_df.iterrows():
            diff_parts = []
            for param in differing_params:
                val = row.get(param)
                if val is not None:
                    param_name = param.split(".")[-1]
                    diff_parts.append(f"{param_name}={display_value(val)}")
            run_labels[run_name] = ", ".join(diff_parts)
    else:
        for run_name in config_df.index:
            run_labels[run_name] = run_name

    config_table = Table(
        title="[bold cyan]Configuration Differences[/bold cyan]",
        show_header=True,
        header_style="bold magenta",
    )
    config_table.add_column("Run Label", style="dim", width=30)
    for run_name in sorted(config_df.index, key=lambda x: run_labels[x]):
        config_table.add_column(run_labels[run_name], justify="right")

    for param in differing_params:
        values = [
            display_value(config_df.loc[run_name, param])
            for run_name in sorted(config_df.index, key=lambda x: run_labels[x])
        ]
        config_table.add_row(param, *values)

    if differing_params:
        console.print(Panel(config_table, border_style="yellow", expand=False))
    else:
        console.print("[bold yellow]No configuration differences found among runs.[/bold yellow]")

    # Results comparison
    combined_df = pd.concat(all_dfs, ignore_index=True)
    combined_df["run_label"] = combined_df["run_label"].map(run_labels)

    console.print(create_comparison_summary_panel(combined_df))

    console.print(f"Saving comparison plots to [bold]{output_dir}[/bold]")

    solved_df = combined_df[combined_df["solved"]].copy()
    if not solved_df.empty:
        sorted_labels = sorted(combined_df["run_label"].unique())

        expected_varying_cols = ["pop_ratio", "cost_weight", "batch_size"]
        varying_cols = [p.split(".")[-1] for p in differing_params if p.startswith("eval_options.")]
        varying_cols = [c for c in varying_cols if c in expected_varying_cols]

        plots = plot_comparison_analysis(
            solved_df,
            sorted_labels,
            scatter_max_points=scatter_max_points,
            varying_params=varying_cols,
        )
        for plot_name, fig in plots.items():
            fig.savefig(output_dir / f"{plot_name}.png")
            plt.close(fig)


@evaluation.command(name="compare")
@click.argument("run_dirs", nargs=-1, type=click.Path(exists=True, file_okay=False))
@click.option(
    "--scatter-max-points",
    type=int,
    default=2000,
    help="Maximum number of points to display on scatter plots.",
)
def eval_compare(run_dirs: list[str], scatter_max_points: int):
    """
    Compare multiple evaluation runs.

    Can accept individual run directories or parent directories containing multiple runs.
    """
    console = Console()
    if not run_dirs:
        console.print("[bold red]Error: Please provide at least one run directory.[/bold red]")
        return

    # Discover all individual run directories
    actual_run_dirs = []
    for run_dir_str in run_dirs:
        run_dir = Path(run_dir_str)
        # Check if the directory itself is a run directory
        if (run_dir / "results.csv").exists():
            actual_run_dirs.append(run_dir_str)
        else:
            # If not, treat it as a parent and search for sub-directories
            sub_dirs_found = [
                str(sub_dir)
                for sub_dir in run_dir.iterdir()
                if sub_dir.is_dir() and (sub_dir / "results.csv").exists()
            ]
            if sub_dirs_found:
                console.print(f"Found {len(sub_dirs_found)} sub-runs in [bold]{run_dir}[/bold]")
                actual_run_dirs.extend(sub_dirs_found)
            else:
                console.print(
                    f"[yellow]Warning: Directory {run_dir} is not a valid run and contains no sub-runs."
                    f"Skipping.[/yellow]"
                )

    if not actual_run_dirs:
        console.print("[bold red]Error: No valid run directories found to compare.[/bold red]")
        return

    # Determine output directory
    output_dir: Path
    if len(run_dirs) == 1 and Path(run_dirs[0]).is_dir():
        # If a single directory is provided (likely a sweep parent), save results there
        output_dir = Path(run_dirs[0])
    else:
        # For multiple dirs or other cases, create a new comparison directory
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        output_dir = Path("runs") / f"comparison_{timestamp}"

    output_dir.mkdir(parents=True, exist_ok=True)

    # Use set to avoid duplicates if user provides both parent and sub-run
    unique_run_dirs = sorted(list(set(actual_run_dirs)))

    _generate_comparison_report(
        run_dirs=unique_run_dirs,
        output_dir=output_dir,
        scatter_max_points=scatter_max_points,
    )
    console.print(f"Comparison report saved in [bold]{output_dir}[/bold]")
