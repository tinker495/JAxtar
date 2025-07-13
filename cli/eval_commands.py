import json
import os
import time
from collections.abc import MutableMapping
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
from xtructure import xtructure_numpy as xnp

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


def log_and_summarize_evaluation(
    results: list[dict], run_dir: Path, console: Console, config: dict
):
    """Logs evaluation results and config to files and prints a summary."""
    run_dir.mkdir(parents=True, exist_ok=True)
    console.print(f"Saving results to [bold]{run_dir}[/bold]")

    # Save config
    with open(run_dir / "config.json", "w") as f:
        json.dump(config, f, indent=4)

    os.makedirs(run_dir / "path_states", exist_ok=True)
    for r in results:
        if r.get("path_analysis"):
            states = r["path_analysis"]["states"]
            states.save(run_dir / "path_states" / f"{r['seed']}.npz")

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

    # Scatter plot for pop_ratio comparison
    if "pop_ratio" in df.columns and df["pop_ratio"].nunique() > 1:
        plt.figure(figsize=(12, 8))
        sns.scatterplot(
            data=df[df["solved"]],
            x="nodes_generated",
            y="search_time_s",
            hue="pop_ratio",
            palette="viridis",
            alpha=0.7,
            edgecolor=None,
        )
        plt.xscale("log")
        plt.yscale("log")
        plt.title("Search Time vs. Generated Nodes by Pop Ratio")
        plt.xlabel("Nodes Generated (log scale)")
        plt.ylabel("Search Time (s, log scale)")
        plt.legend(title="Pop Ratio", bbox_to_anchor=(1.05, 1), loc="upper left")
        plt.grid(True, which="both", linestyle="--", linewidth=0.7, alpha=0.6)
        plt.tight_layout()
        plt.savefig(run_dir / "nodes_vs_time_pop_ratio_scatter.png")
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


def summarize_and_plot_popratios(results, run_dir, console):
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    import seaborn as sns
    from rich.panel import Panel
    from rich.table import Table

    from helpers import human_format

    df = pd.DataFrame(results)
    summary_table = Table(
        title="[bold cyan]Pop Ratio Comparison Summary[/bold cyan]",
        show_header=True,
        header_style="bold magenta",
    )
    summary_table.add_column("Pop Ratio", style="dim", width=15)
    summary_table.add_column("Success Rate", justify="right")
    summary_table.add_column("Avg. Search Time (Solved)", justify="right")
    summary_table.add_column("Avg. Generated Nodes (Solved)", justify="right")
    summary_table.add_column("Avg. Path Cost", justify="right")

    for pr, group in df.groupby("pop_ratio"):
        num_puzzles = len(group)
        solved_group = group[group["solved"]]
        num_solved = len(solved_group)
        success_rate = (num_solved / num_puzzles) * 100 if num_puzzles > 0 else 0
        if not solved_group.empty:
            avg_time = solved_group["search_time_s"].mean()
            avg_nodes = solved_group["nodes_generated"].mean()
            avg_path_cost = solved_group["path_cost"].mean()
            summary_table.add_row(
                str(pr),
                f"{success_rate:.2f}% ({num_solved}/{num_puzzles})",
                f"{avg_time:.3f} s",
                human_format(avg_nodes),
                f"{avg_path_cost:.2f}",
            )
        else:
            summary_table.add_row(
                str(pr),
                f"{success_rate:.2f}% ({num_solved}/{num_puzzles})",
                "N/A",
                "N/A",
                "N/A",
            )
    console.print(Panel(summary_table, border_style="green", expand=False))

    solved_df = df[df["solved"]].copy()
    if not solved_df.empty:
        # Convert pop_ratio to string for categorical coloring, handle inf
        if "pop_ratio" in solved_df.columns:
            solved_df["pop_ratio_str"] = (
                solved_df["pop_ratio"].replace([np.inf, -np.inf], "inf").astype(str)
            )
            hue_col = "pop_ratio_str"
        else:
            hue_col = "pop_ratio"

        plt.figure(figsize=(12, 8))
        sns.boxplot(data=solved_df, x="pop_ratio_str", y="path_cost")
        plt.title("Path Cost by Pop Ratio")
        plt.xlabel("Pop Ratio")
        plt.ylabel("Path Cost")
        plt.tight_layout()
        plt.savefig(run_dir / "path_cost_by_popratio.png")
        plt.close()

        plt.figure(figsize=(12, 8))
        sns.boxplot(data=solved_df, x="pop_ratio_str", y="search_time_s")
        plt.yscale("log")
        plt.title("Search Time by Pop Ratio")
        plt.xlabel("Pop Ratio")
        plt.ylabel("Search Time (s, log scale)")
        plt.tight_layout()
        plt.savefig(run_dir / "search_time_by_popratio.png")
        plt.close()

        plt.figure(figsize=(12, 8))
        sns.boxplot(data=solved_df, x="pop_ratio_str", y="nodes_generated")
        plt.title("Generated Nodes by Pop Ratio")
        plt.xlabel("Pop Ratio")
        plt.ylabel("Nodes Generated")
        plt.tight_layout()
        plt.savefig(run_dir / "nodes_generated_by_popratio.png")
        plt.close()

        plt.figure(figsize=(12, 8))
        sns.scatterplot(
            data=solved_df,
            x="nodes_generated",
            y="search_time_s",
            hue=hue_col,
            palette="tab10",
            alpha=0.7,
            edgecolor=None,
        )
        plt.xscale("log")
        plt.yscale("log")
        plt.title("Search Time vs. Generated Nodes by Pop Ratio")
        plt.xlabel("Nodes Generated (log scale)")
        plt.ylabel("Search Time (s, log scale)")
        plt.legend(title="Pop Ratio", bbox_to_anchor=(1.05, 1), loc="upper left")
        plt.grid(True, which="both", linestyle="--", linewidth=0.7, alpha=0.6)
        plt.tight_layout()
        plt.savefig(run_dir / "nodes_vs_time_popratio_scatter.png")
        plt.close()


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
    heuristic_metadata = heuristic.metadata if hasattr(heuristic, "metadata") else {}

    pop_ratios = eval_options.pop_ratio
    if not isinstance(pop_ratios, list):
        pop_ratios = [pop_ratios]

    config = {
        "puzzle": {"name": puzzle_name, "size": puzzle.size},
        "search_algorithm": "A*",
        "heuristic": heuristic.__class__.__name__,
        "heuristic_metadata": heuristic_metadata,
        "eval_options": eval_options.dict(),
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
        # Optionally, save per-pop_ratio results
        pd.DataFrame(results).to_csv(
            run_dir / f"results_pr_{str(pr).replace('.', '_')}.csv", index=False
        )

    # Save combined results and config
    pd.DataFrame(all_results).to_csv(run_dir / "results.csv", index=False)
    with open(run_dir / "config.json", "w") as f:
        json.dump(config, f, indent=4)
    # Save path_states if present
    os.makedirs(run_dir / "path_states", exist_ok=True)
    for r in all_results:
        if r.get("path_analysis"):
            states = r["path_analysis"]["states"]
            states.save(
                run_dir
                / "path_states"
                / f"{r['seed']}_pr_{str(r['pop_ratio']).replace('.', '_')}.npz"
            )
    # Print config
    print_config("Heuristic Evaluation Configuration", config)
    # Pop ratio summary and plots
    summarize_and_plot_popratios(all_results, run_dir, console)


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
    qfunction_metadata = qfunction.metadata if hasattr(qfunction, "metadata") else {}

    pop_ratios = eval_options.pop_ratio
    if not isinstance(pop_ratios, list):
        pop_ratios = [pop_ratios]

    config = {
        "puzzle": {"name": puzzle_name, "size": puzzle.size},
        "search_algorithm": "Q*",
        "q_function": qfunction.__class__.__name__,
        "qfunction_metadata": qfunction_metadata,
        "eval_options": eval_options.dict(),
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
        # Optionally, save per-pop_ratio results
        pd.DataFrame(results).to_csv(
            run_dir / f"results_pr_{str(pr).replace('.', '_')}.csv", index=False
        )

    # Save combined results and config
    pd.DataFrame(all_results).to_csv(run_dir / "results.csv", index=False)
    with open(run_dir / "config.json", "w") as f:
        json.dump(config, f, indent=4)
    # Save path_states if present
    os.makedirs(run_dir / "path_states", exist_ok=True)
    for r in all_results:
        if r.get("path_analysis"):
            states = r["path_analysis"]["states"]
            states.save(
                run_dir
                / "path_states"
                / f"{r['seed']}_pr_{str(r['pop_ratio']).replace('.', '_')}.npz"
            )
    # Print config
    print_config("Q-Learning Evaluation Configuration", config)
    # Pop ratio summary and plots
    summarize_and_plot_popratios(all_results, run_dir, console)


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
def eval_compare(run_dirs: list[str]):
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

    # Config comparison (as before)
    flat_configs = {name: flatten_dict(cfg) for name, cfg in all_configs.items()}
    config_df = pd.DataFrame.from_dict(flat_configs, orient="index")
    differing_params = []
    if not config_df.empty:
        for col in sorted(config_df.columns):
            try:
                # Convert all values in the column to hashable before nunique
                if config_df[col].apply(make_hashable).nunique() > 1:
                    differing_params.append(col)
            except Exception as e:
                print(f"Skipping column {col} due to error: {e}")
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

    # Build composite run_labels including all differing config params
    run_labels = {}
    for label in composite_labels:
        run_name = label_to_run[label]
        pop_ratio = label_to_pop[label]
        diff_parts = []
        for param in differing_params:
            # Skip any config param that is a list of pop_ratios (not the actual group value)
            if "pop_ratio" in param:
                continue
            val = config_df.loc[run_name, param]
            diff_parts.append(f"{param.split('.')[-1]}={display_value(val)}")
        # Always add the actual pop_ratio for this group, if available
        if pop_ratio is not None:
            diff_parts.append(f"pop_ratio={pop_ratio}")
        if diff_parts:
            run_labels[label] = ", ".join(diff_parts)
        else:
            run_labels[label] = "default"
    # Apply new run_label to combined_df
    combined_df["run_label"] = combined_df["run_label"].map(run_labels)

    summary_table = Table(
        title="[bold cyan]Evaluation Comparison Summary[/bold cyan]",
        show_header=True,
        header_style="bold magenta",
    )
    summary_table.add_column("Run (Config Differences)", style="dim", width=30)
    summary_table.add_column("Success Rate", justify="right")
    summary_table.add_column("Avg. Search Time (Solved)", justify="right")
    summary_table.add_column("Avg. Generated Nodes (Solved)", justify="right")
    summary_table.add_column("Avg. Path Cost", justify="right")

    for label, group in combined_df.groupby("run_label"):
        num_puzzles = len(group)
        solved_group = group[group["solved"]]
        num_solved = len(solved_group)
        success_rate = (num_solved / num_puzzles) * 100 if num_puzzles > 0 else 0
        if not solved_group.empty:
            avg_time = solved_group["search_time_s"].mean()
            avg_nodes = solved_group["nodes_generated"].mean()
            avg_path_cost = solved_group["path_cost"].mean()
            summary_table.add_row(
                label,
                f"{success_rate:.2f}% ({num_solved}/{num_puzzles})",
                f"{avg_time:.3f} s",
                human_format(avg_nodes),
                f"{avg_path_cost:.2f}",
            )
        else:
            summary_table.add_row(
                label,
                f"{success_rate:.2f}% ({num_solved}/{num_puzzles})",
                "N/A",
                "N/A",
                "N/A",
            )
    console.print(Panel(summary_table, border_style="green", expand=False))

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    comparison_dir = Path("runs") / f"comparison_{timestamp}"
    comparison_dir.mkdir(parents=True, exist_ok=True)
    console.print(f"Saving comparison plots to [bold]{comparison_dir}[/bold]")

    solved_df = combined_df[combined_df["solved"]].copy()
    if not solved_df.empty:
        sorted_labels = [
            run_labels[composite_label]
            for composite_label in composite_labels
            if run_labels[composite_label] in solved_df["run_label"].unique()
        ]
        plt.figure(figsize=(12, 8))
        sns.boxplot(data=solved_df, x="run_label", y="path_cost", order=sorted_labels)
        plt.title("Path Cost Comparison")
        plt.xlabel("Run (Config Differences)")
        plt.ylabel("Path Cost")
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()
        plt.savefig(comparison_dir / "path_cost_comparison.png")
        plt.close()

        plt.figure(figsize=(12, 8))
        sns.boxplot(data=solved_df, x="run_label", y="search_time_s", order=sorted_labels)
        plt.yscale("log")
        plt.title("Search Time Comparison")
        plt.xlabel("Run (Config Differences)")
        plt.ylabel("Search Time (s, log scale)")
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()
        plt.savefig(comparison_dir / "search_time_comparison.png")
        plt.close()

        plt.figure(figsize=(12, 8))
        sns.boxplot(data=solved_df, x="run_label", y="nodes_generated", order=sorted_labels)
        plt.title("Generated Nodes Comparison")
        plt.xlabel("Run (Config Differences)")
        plt.ylabel("Nodes Generated")
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()
        plt.savefig(comparison_dir / "nodes_generated_comparison.png")
        plt.close()

        plt.figure(figsize=(12, 8))
        sns.scatterplot(
            data=solved_df,
            x="nodes_generated",
            y="search_time_s",
            hue="run_label",
            palette="tab10",
            alpha=0.7,
            edgecolor=None,
        )
        plt.xscale("log")
        plt.yscale("log")
        plt.title("Search Time vs. Generated Nodes (Scatter)")
        plt.xlabel("Nodes Generated (log scale)")
        plt.ylabel("Search Time (s, log scale)")
        plt.legend(title="Run (Config Differences)", bbox_to_anchor=(1.05, 1), loc="upper left")
        plt.tight_layout()
        plt.grid(True, which="both", linestyle="--", linewidth=0.7, alpha=0.6)
        plt.savefig(comparison_dir / "nodes_vs_time_scatter.png")
        plt.close()
