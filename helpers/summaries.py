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
