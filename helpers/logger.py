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
            solved_paths = [r["path_length"] for r in solved_results]

            self.log_scalar(
                "Evaluation/Avg Search Time (Solved)", jnp.mean(jnp.array(solved_times)), step
            )
            self.log_scalar(
                "Evaluation/Avg Generated Nodes (Solved)", jnp.mean(jnp.array(solved_nodes)), step
            )
            self.log_scalar("Evaluation/Avg Path Length", jnp.mean(jnp.array(solved_paths)), step)
        else:
            self.log_scalar("Evaluation/Avg Search Time (Solved)", 0, step)
            self.log_scalar("Evaluation/Avg Generated Nodes (Solved)", 0, step)
            self.log_scalar("Evaluation/Avg Path Length", 0, step)

        solved_df = df[df["solved"]]
        if not solved_df.empty:
            fig = plt.figure(figsize=(10, 6))
            sns.histplot(data=solved_df, x="search_time_s", kde=True)
            plt.title("Distribution of Search Time")
            self.writer.add_figure("Evaluation/Plots/Search Time Distribution", fig, step)
            plt.close()

            fig = plt.figure(figsize=(10, 6))
            sns.histplot(data=solved_df, x="nodes_generated", kde=True)
            plt.title("Distribution of Generated Nodes")
            self.writer.add_figure("Evaluation/Plots/Nodes Generated Distribution", fig, step)
            plt.close()

            fig = plt.figure(figsize=(10, 6))
            sns.histplot(data=solved_df, x="path_length", kde=True)
            plt.title("Distribution of Path Length")
            self.writer.add_figure("Evaluation/Plots/Path Length Distribution", fig, step)
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
            self.writer.add_figure("Evaluation/Plots/Search Time by Path Length", fig, step)
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
            self.writer.add_figure("Evaluation/Plots/Generated Nodes by Path Length", fig, step)
            plt.close()

    def close(self):
        self.writer.close()
        if self.aim_run:
            self.aim_run.close()
