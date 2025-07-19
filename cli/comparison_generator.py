import json
from pathlib import Path
from typing import List, Optional

import matplotlib.pyplot as plt
import pandas as pd
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from helpers.logger import TensorboardLogger
from helpers.plots import plot_comparison_analysis
from helpers.summaries import create_comparison_summary_panel
from helpers.util import display_value, flatten_dict, make_hashable


class ComparisonGenerator:
    def __init__(
        self,
        run_dirs: List[str],
        output_dir: Path,
        scatter_max_points: int,
        logger: Optional[TensorboardLogger] = None,
        step: int = 0,
    ):
        self.run_dirs = run_dirs
        self.output_dir = output_dir
        self.scatter_max_points = scatter_max_points
        self.logger = logger
        self.step = step
        self.console = Console()

    def generate_report(self):
        all_dfs = []
        all_configs = {}
        run_labels = {}

        for run_dir_str in self.run_dirs:
            run_dir = Path(run_dir_str)
            run_name = run_dir.name

            results_path = run_dir / "results.csv"
            if not results_path.exists():
                self.console.print(
                    f"[bold red]Warning: Cannot find results.csv in {run_dir_str}. Skipping.[/bold red]"
                )
                continue

            config_path = run_dir / "config.json"
            if not config_path.exists():
                self.console.print(
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
            self.console.print("[bold red]No valid evaluation runs found to compare.[/bold red]")
            return

        # Config comparison
        flat_configs = {name: flatten_dict(cfg) for name, cfg in all_configs.items()}
        config_df = pd.DataFrame.from_dict(flat_configs, orient="index")
        differing_params = []
        if not config_df.empty:
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
                    and "heuristic_metrics" not in col
                ]
            )

            for col in priority_params + other_params:
                if col not in config_df.columns:
                    continue
                try:
                    if config_df[col].apply(make_hashable).nunique() > 1:
                        differing_params.append(col)
                except Exception as e:
                    self.console.print(f"Skipping column {col} due to error: {e}", style="dim red")

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
            self.console.print(Panel(config_table, border_style="yellow", expand=False))
        else:
            self.console.print(
                "[bold yellow]No configuration differences found among runs.[/bold yellow]"
            )

        # Results comparison
        combined_df = pd.concat(all_dfs, ignore_index=True)

        if "heuristic_metrics.r_squared" in config_df.columns:
            combined_df = combined_df.merge(
                config_df[["heuristic_metrics.r_squared", "heuristic_metrics.ccc"]],
                left_on="run_label",
                right_index=True,
                how="left",
            )

        combined_df["run_label"] = combined_df["run_label"].map(run_labels)

        self.console.print(create_comparison_summary_panel(combined_df))
        self.console.print(f"Saving comparison plots to [bold]{self.output_dir}[/bold]")

        solved_df = combined_df[combined_df["solved"]].copy()
        if not solved_df.empty:
            sorted_labels = sorted(combined_df["run_label"].unique())

            expected_varying_cols = ["pop_ratio", "cost_weight", "batch_size"]
            varying_cols = [
                p.split(".")[-1] for p in differing_params if p.startswith("eval_options.")
            ]
            varying_cols = [c for c in varying_cols if c in expected_varying_cols]

            plots = plot_comparison_analysis(
                solved_df,
                sorted_labels,
                scatter_max_points=self.scatter_max_points,
                varying_params=varying_cols,
            )
            for plot_name, fig in plots.items():
                fig.savefig(self.output_dir / f"{plot_name}.png")
                if self.logger:
                    self.logger.log_figure(f"Comparison/{plot_name}", fig, self.step)
                plt.close(fig)
