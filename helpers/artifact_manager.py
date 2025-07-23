import json
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import pandas as pd

from helpers.logger import BaseLogger
from helpers.util import convert_to_serializable_dict


class ArtifactManager:
    def __init__(self, run_dir: Path, logger: Optional[BaseLogger] = None, step: int = 0):
        self.run_dir = run_dir
        self.logger = logger
        self.step = step
        self.run_dir.mkdir(parents=True, exist_ok=True)

    def save_config(self, config: dict):
        """Saves the configuration dictionary to a JSON file."""
        with open(self.run_dir / "config.json", "w") as f:
            serializable_config = convert_to_serializable_dict(config)
            json.dump(serializable_config, f, indent=4)

    def save_results(self, results: List[Dict]):
        """Saves the main evaluation results to a CSV file."""
        pd.DataFrame(results).to_csv(self.run_dir / "results.csv", index=False)

    def save_path_states(self, results: List[Dict]):
        """Saves path states to .npz files if they exist in the results."""
        path_states_dir = self.run_dir / "path_states"
        path_states_dir.mkdir(exist_ok=True)
        for r in results:
            if r.get("path_analysis") and "states" in r["path_analysis"]:
                states = r["path_analysis"]["states"]
                filename = f"{r['seed']}.npz"
                states.save(path_states_dir / filename)

    def save_and_log_plot(self, plot_name: str, fig: plt.Figure, sub_dir: Optional[str] = None):
        """Saves a plot to a file and logs it to Tensorboard if a logger is available."""
        plot_dir = self.run_dir
        if sub_dir:
            plot_dir = self.run_dir / sub_dir
            plot_dir.mkdir(exist_ok=True)

        filepath = plot_dir / f"{plot_name}.png"
        fig.savefig(filepath)

        if self.logger:
            log_tag = f"Comparison/{self.run_dir.name}/{plot_name}"
            self.logger.log_figure(log_tag, fig, self.step)

        plt.close(fig)

    def log_scalar(self, name: str, value: float):
        """Logs a scalar value to Tensorboard if a logger is available."""
        if self.logger:
            log_tag = f"Comparison/{self.run_dir.name}/{name}"
            self.logger.log_scalar(log_tag, value, self.step)
