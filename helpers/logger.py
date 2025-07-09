import os
import subprocess
from datetime import datetime

import aim
import numpy as np
import tensorboardX


class TensorboardLogger:
    def __init__(self, log_dir_base: str, config: dict):
        self.config = config
        self.log_dir = self._create_log_dir(log_dir_base)
        self.writer = tensorboardX.SummaryWriter(self.log_dir)

        # Initialize Aim run
        self.aim_run = aim.Run(
            experiment=log_dir_base,
        )

        self.log_hyperparameters()
        self.log_git_info()
        print(f"Tensorboard log directory: {self.log_dir}")
        print(f"Aim repo location: {self.aim_run.repo.path}")
        print(f"Aim run hash: {self.aim_run.hash}")

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
        self.aim_run["hparams"] = self.config

    def log_git_info(self):
        try:
            commit_hash = (
                subprocess.check_output(["git", "rev-parse", "HEAD"]).strip().decode("utf-8")
            )
            self.writer.add_text("Git Commit", commit_hash)
            self.aim_run["git_commit"] = commit_hash
        except (subprocess.CalledProcessError, FileNotFoundError):
            self.writer.add_text("Git Commit", "N/A")
            self.aim_run["git_commit"] = "N/A"

    def log_scalar(self, tag: str, value: float, step: int):
        self.writer.add_scalar(tag, value, step)
        self.aim_run.track(float(value), name=tag, step=step)

    def log_histogram(self, tag: str, values: np.ndarray, step: int):
        self.writer.add_histogram(tag, values, step)
        # Aim doesn't have a direct histogram equivalent, but we can log distributions
        # For simplicity, we can log mean/std/min/max or just skip it.
        # Let's log a distribution for now.
        self.aim_run.track(aim.Distribution(values), name=tag, step=step)

    def log_image(self, tag: str, image: np.ndarray, step: int, dataformats="HWC"):
        self.writer.add_image(tag, image, step, dataformats=dataformats)
        # Aim's Image requires channel-first format (CHW)
        if dataformats == "HWC":
            aim_image = np.transpose(image, (2, 0, 1))
        else:
            aim_image = image
        self.aim_run.track(aim.Image(aim_image), name=tag, step=step)

    def log_text(self, tag: str, text: str, step: int = 0):
        self.writer.add_text(tag, text, step)
        self.aim_run.track(aim.Text(text), name=tag, step=step)

    def close(self):
        self.writer.close()
        self.aim_run.close()
