from typing import Dict, Optional

import jax.numpy as jnp
import pandas as pd
from rich.panel import Panel
from rich.table import Table

from helpers.formatting import human_format


def create_summary_panel(results: list[dict], metrics: Optional[Dict[str, float]] = None) -> Panel:
    """Creates a rich Panel with a summary of evaluation results."""
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

    if metrics:
        r_squared = metrics.get("r_squared")
        ccc = metrics.get("ccc")
        if r_squared is not None:
            summary_table.add_row("Heuristic R-squared (R²)", f"{r_squared:.3f}")
        if ccc is not None:
            summary_table.add_row("Heuristic CCC (ρc)", f"{ccc:.3f}")

    if solved_results:
        solved_times = [r["search_time_s"] for r in solved_results]
        solved_nodes = [r["nodes_generated"] for r in solved_results]
        solved_paths = [r["path_cost"] for r in solved_results]

        summary_table.add_row(
            "Avg. Search Time (Solved)", f"{jnp.mean(jnp.array(solved_times)):.3f} s"
        )
        summary_table.add_row(
            "Avg. Generated Nodes (Solved)",
            human_format(jnp.mean(jnp.array(solved_nodes))),
        )
        summary_table.add_row("Avg. Path Cost", f"{jnp.mean(jnp.array(solved_paths)):.2f}")

        optimal_costs = [r.get("benchmark_optimal_path_cost") for r in solved_results]
        optimal_costs = [c for c in optimal_costs if c is not None]
        if optimal_costs:
            mean_opt_cost = jnp.mean(jnp.array(optimal_costs))
            summary_table.add_row("Avg. Optimal Cost", f"{float(mean_opt_cost):.2f}")
            cost_gaps = [
                r["path_cost"] - r["benchmark_optimal_path_cost"]
                for r in solved_results
                if r.get("benchmark_optimal_path_cost") is not None
            ]
            if cost_gaps:
                mean_gap = jnp.mean(jnp.array(cost_gaps))
                summary_table.add_row("Avg. Cost Gap", f"{float(mean_gap):+.2f}")

        path_action_counts = [r.get("path_action_count") for r in solved_results]
        path_action_counts = [c for c in path_action_counts if c is not None]
        optimal_action_counts = [r.get("benchmark_optimal_action_count") for r in solved_results]
        optimal_action_counts = [c for c in optimal_action_counts if c is not None]
        if path_action_counts:
            mean_actions = jnp.mean(jnp.array(path_action_counts))
            summary_table.add_row("Avg. Solution Length (actions)", f"{float(mean_actions):.2f}")
        if optimal_action_counts:
            mean_opt_actions = jnp.mean(jnp.array(optimal_action_counts))
            summary_table.add_row("Avg. Optimal Length (actions)", f"{float(mean_opt_actions):.2f}")
            if path_action_counts:
                length_pairs = [
                    (r.get("path_action_count"), r.get("benchmark_optimal_action_count"))
                    for r in solved_results
                    if r.get("path_action_count") is not None
                    and r.get("benchmark_optimal_action_count") is not None
                ]
                if length_pairs:
                    gaps = [actual - optimal for actual, optimal in length_pairs]
                    mean_length_gap = jnp.mean(jnp.array(gaps))
                    summary_table.add_row(
                        "Avg. Length Gap (actions)", f"{float(mean_length_gap):+.2f}"
                    )

        match_flags = [r.get("matches_optimal_path") for r in solved_results]
        match_flags = [m for m in match_flags if m is not None]
        if match_flags:
            exact_match_rate = sum(1 for m in match_flags if m) / len(match_flags)
            summary_table.add_row(
                "Exact Optimal Paths",
                f"{exact_match_rate * 100:.1f}% ({sum(1 for m in match_flags if m)}/{len(match_flags)})",
            )
    else:
        summary_table.add_row("Avg. Search Time (Solved)", "N/A")
        summary_table.add_row("Avg. Generated Nodes (Solved)", "N/A")
        summary_table.add_row("Avg. Path Cost", "N/A")

    return Panel(summary_table, border_style="green", expand=False)


def create_comparison_summary_panel(combined_df: pd.DataFrame) -> Panel:
    """Creates a rich Panel with a summary of comparison results."""
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
    summary_table.add_column("R²", justify="right")
    summary_table.add_column("CCC", justify="right")

    for label, group in combined_df.groupby("run_label"):
        num_puzzles = len(group)
        solved_group = group[group["solved"]]
        num_solved = len(solved_group)
        success_rate = (num_solved / num_puzzles) * 100 if num_puzzles > 0 else 0

        # Heuristic metrics are stored per-run, not per-row. We need to get them from the config.
        # This requires a bit of a workaround since the config isn't directly passed here.
        # We assume the comparison report generation step will add these to the dataframe.
        r_squared = None
        if "heuristic_metrics.r_squared" in group.columns:
            r_squared = group["heuristic_metrics.r_squared"].iloc[0]
        ccc = None
        if "heuristic_metrics.ccc" in group.columns:
            ccc = group["heuristic_metrics.ccc"].iloc[0]

        r_squared_str = f"{r_squared:.3f}" if pd.notna(r_squared) else "N/A"
        ccc_str = f"{ccc:.3f}" if pd.notna(ccc) else "N/A"

        if not solved_group.empty:
            avg_time_str = f"{solved_group['search_time_s'].mean():.3f} s"
            avg_nodes_str = human_format(solved_group["nodes_generated"].mean())
            avg_path_cost_str = f"{solved_group['path_cost'].mean():.2f}"
        else:
            avg_time_str = "N/A"
            avg_nodes_str = "N/A"
            avg_path_cost_str = "N/A"

        summary_table.add_row(
            label,
            f"{success_rate:.2f}% ({num_solved}/{num_puzzles})",
            avg_time_str,
            avg_nodes_str,
            avg_path_cost_str,
            r_squared_str,
            ccc_str,
        )
    return Panel(summary_table, border_style="green", expand=False)
