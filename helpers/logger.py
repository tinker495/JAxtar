import os
import subprocess
from datetime import datetime
from typing import Any

import aim
import numpy as np
import tensorboardX
from pydantic import BaseModel


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

    def close(self):
        self.writer.close()
        if self.aim_run:
            self.aim_run.close()
