import jax.numpy as jnp
import pandas as pd
from rich.panel import Panel
from rich.table import Table

from helpers.formatting import human_format


def create_summary_panel(results: list[dict]) -> Panel:
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


def create_pop_ratio_summary_panel(results: list[dict]) -> Panel:
    """Creates a rich Panel with a summary of pop ratio comparison results."""
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
    return Panel(summary_table, border_style="green", expand=False)
