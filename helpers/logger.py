import os
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Any

import aim
import jax.numpy as jnp
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import tensorboardX
from pydantic import BaseModel

matplotlib.use("Agg")


def _convert_to_dict_if_pydantic(obj: Any) -> Any:
    if isinstance(obj, BaseModel):
        return obj.dict()
    if isinstance(obj, dict):
        return {k: _convert_to_dict_if_pydantic(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_convert_to_dict_if_pydantic(i) for i in obj]
    return obj


class TensorboardLogger:
    def __init__(self, log_dir_base: str, config: dict):
        self.config = config
        self.log_dir = self._create_log_dir(log_dir_base)
        self.writer = tensorboardX.SummaryWriter(self.log_dir)

        # Initialize Aim run
        self.aim_run = None
        try:
            self.aim_run = aim.Run(
                experiment=log_dir_base,
            )
            print(f"Aim logging enabled. Repo: {self.aim_run.repo.path}")
            print(f"Aim run hash: {self.aim_run.hash}")
        except Exception as e:
            print(f"Could not initialize Aim, disabling Aim logging. Error: {e}")

        self.log_hyperparameters()
        self.log_git_info()
        print(f"Tensorboard log directory: {self.log_dir}")

    def _create_log_dir(self, log_dir_base: str) -> str:
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        log_dir = os.path.join("runs", f"{log_dir_base}_{timestamp}")
        os.makedirs(log_dir, exist_ok=True)
        return log_dir

    def log_hyperparameters(self):
        # Using add_text to keep it simple and viewable. add_hparams is more complex.
        config_str = "\n".join([f"{key}: {value}" for key, value in self.config.items()])
        self.writer.add_text("Configuration", config_str)

        # also save config to a file
        with open(os.path.join(self.log_dir, "config.txt"), "w") as f:
            f.write(config_str)

        # Log hyperparameters to Aim
        if self.aim_run:
            hparams = _convert_to_dict_if_pydantic(self.config)
            self.aim_run["hparams"] = hparams

    def log_git_info(self):
        try:
            commit_hash = (
                subprocess.check_output(["git", "rev-parse", "HEAD"]).strip().decode("utf-8")
            )
            self.writer.add_text("Git Commit", commit_hash)
            if self.aim_run:
                self.aim_run["git_commit"] = commit_hash
        except (subprocess.CalledProcessError, FileNotFoundError):
            self.writer.add_text("Git Commit", "N/A")
            if self.aim_run:
                self.aim_run["git_commit"] = "N/A"

    def log_scalar(self, tag: str, value: float, step: int):
        self.writer.add_scalar(tag, value, step)
        if self.aim_run:
            self.aim_run.track(float(value), name=tag, step=step)

    def log_histogram(self, tag: str, values: np.ndarray, step: int):
        self.writer.add_histogram(tag, values, step)
        if self.aim_run:
            # Aim doesn't have a direct histogram equivalent, but we can log distributions
            # For simplicity, we can log mean/std/min/max or just skip it.
            # Let's log a distribution for now.
            self.aim_run.track(aim.Distribution(values), name=tag, step=step)

    def log_image(self, tag: str, image, step: int, dataformats="HWC"):
        image = np.asarray(image)
        self.writer.add_image(tag, image, step, dataformats=dataformats)
        if self.aim_run:
            # Aim's Image requires channel-first format (CHW)
            if dataformats == "CHW":
                aim_image = np.transpose(image, (2, 0, 1))
            else:
                aim_image = image
            self.aim_run.track(aim.Image(aim_image), name=tag, step=step)

    def log_text(self, tag: str, text: str, step: int = 0):
        self.writer.add_text(tag, text, step)
        if self.aim_run:
            self.aim_run.track(aim.Text(text), name=tag, step=step)

    def log_figure(self, tag: str, figure, step: int):
        """Logs a Matplotlib figure to TensorBoard and Aim."""
        self.writer.add_figure(tag, figure, step)
        if self.aim_run:
            self.aim_run.track(aim.Figure(figure), name=tag, step=step)

    def log_evaluation_results(self, results: list[dict], step: int):
        """Logs evaluation results to files and logs summary to TensorBoard/Aim."""
        run_dir = Path(self.log_dir)
        df = pd.DataFrame(results)
        df.to_csv(run_dir / "results.csv", index=False)

        num_puzzles = len(results)
        solved_results = [r for r in results if r["solved"]]
        num_solved = len(solved_results)
        success_rate = (num_solved / num_puzzles) * 100 if num_puzzles > 0 else 0

        self.log_scalar("Evaluation/Success Rate", success_rate, step)

        if solved_results:
            solved_times = [r["search_time_s"] for r in solved_results]
            solved_nodes = [r["nodes_generated"] for r in solved_results]
            solved_paths = [r["path_cost"] for r in solved_results]

            self.log_scalar(
                "Evaluation/Avg Search Time (Solved)", jnp.mean(jnp.array(solved_times)), step
            )
            self.log_scalar(
                "Evaluation/Avg Generated Nodes (Solved)", jnp.mean(jnp.array(solved_nodes)), step
            )
            self.log_scalar("Evaluation/Avg Path Cost", jnp.mean(jnp.array(solved_paths)), step)
        else:
            self.log_scalar("Evaluation/Avg Search Time (Solved)", 0, step)
            self.log_scalar("Evaluation/Avg Generated Nodes (Solved)", 0, step)
            self.log_scalar("Evaluation/Avg Path Cost", 0, step)

        solved_df = df[df["solved"]]
        if not solved_df.empty:
            fig = plt.figure(figsize=(10, 6))
            sns.histplot(data=solved_df, x="search_time_s", kde=True)
            plt.title("Distribution of Search Time")
            self.log_figure("Evaluation/Plots/Search Time Distribution", fig, step)
            plt.close(fig)

            fig = plt.figure(figsize=(10, 6))
            sns.histplot(data=solved_df, x="nodes_generated", kde=True)
            plt.title("Distribution of Generated Nodes")
            self.log_figure("Evaluation/Plots/Nodes Generated Distribution", fig, step)
            plt.close(fig)

            fig = plt.figure(figsize=(10, 6))
            sns.histplot(data=solved_df, x="path_cost", kde=True)
            plt.title("Distribution of Path Cost")
            self.log_figure("Evaluation/Plots/Path Cost Distribution", fig, step)
            plt.close(fig)

            fig, ax = plt.subplots(figsize=(10, 6))
            sns.boxplot(
                data=solved_df,
                x="path_cost",
                y="search_time_s",
                ax=ax,
                flierprops=dict(markerfacecolor="silver", markeredgecolor="gray"),
            )
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
            self.log_figure("Evaluation/Plots/Search Time by Path Cost", fig, step)
            plt.close(fig)

            fig, ax = plt.subplots(figsize=(10, 6))
            sns.boxplot(
                data=solved_df,
                x="path_cost",
                y="nodes_generated",
                ax=ax,
                flierprops=dict(markerfacecolor="silver", markeredgecolor="gray"),
            )
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
            self.log_figure("Evaluation/Plots/Generated Nodes by Path Cost", fig, step)
            plt.close(fig)

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

            sns.boxplot(
                data=plot_df,
                x="Actual Cost to Goal",
                y="Estimated Distance",
                ax=ax,
                flierprops=dict(markerfacecolor="silver", markeredgecolor="gray"),
            )
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
                [0, limit],
                [0, limit],
                "g--",
                alpha=0.75,
                zorder=0,
                label="y=x (Perfect Heuristic)",
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
            self.log_figure("Evaluation/Plots/Heuristic Accuracy", fig, step)
            plt.close(fig)

    def close(self):
        self.writer.close()
        if self.aim_run:
            self.aim_run.close()
