import os
import subprocess
from abc import ABC, abstractmethod
from datetime import datetime

import aim
import imageio.v2 as imageio
import matplotlib
import numpy as np
import tensorboardX

from helpers.util import convert_to_serializable_dict

matplotlib.use("Agg")


class BaseLogger(ABC):
    def __init__(self, log_dir_base: str, config: dict):
        self.config = config
        self.log_dir = self._create_log_dir(log_dir_base)
        print(f"Log directory: {self.log_dir}")

    def _create_log_dir(self, log_dir_base: str) -> str:
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        log_dir = os.path.join("runs", f"{log_dir_base}_{timestamp}")
        os.makedirs(log_dir, exist_ok=True)
        return log_dir

    def _log_hyperparameters(self):
        config_str = "\n".join([f"{key}: {value}" for key, value in self.config.items()])
        with open(os.path.join(self.log_dir, "config.txt"), "w") as f:
            f.write(config_str)

    @abstractmethod
    def _log_git_info(self):
        pass

    @abstractmethod
    def log_scalar(self, tag: str, value: float, step: int):
        pass

    @abstractmethod
    def log_histogram(self, tag: str, values: np.ndarray, step: int):
        pass

    @abstractmethod
    def log_image(self, tag: str, image, step: int, dataformats="HWC"):
        pass

    @abstractmethod
    def log_text(self, tag: str, text: str, step: int = 0):
        pass

    @abstractmethod
    def log_figure(self, tag: str, figure, step: int):
        pass

    @abstractmethod
    def close(self):
        pass


class TensorboardLogger(BaseLogger):
    def __init__(self, log_dir_base: str, config: dict):
        super().__init__(log_dir_base, config)
        self.writer = tensorboardX.SummaryWriter(self.log_dir)
        print(f"Tensorboard log directory: {self.log_dir}")
        self._log_hyperparameters()
        self._log_git_info()

    def _log_hyperparameters(self):
        super()._log_hyperparameters()
        config_str = "\n".join([f"{key}: {value}" for key, value in self.config.items()])
        self.writer.add_text("Configuration", config_str)

    def _log_git_info(self):
        try:
            commit_hash = (
                subprocess.check_output(["git", "rev-parse", "HEAD"]).strip().decode("utf-8")
            )
            self.writer.add_text("Git Commit", commit_hash)
        except (subprocess.CalledProcessError, FileNotFoundError):
            self.writer.add_text("Git Commit", "N/A")

    def log_scalar(self, tag: str, value: float, step: int):
        self.writer.add_scalar(tag, value, step)

    def log_histogram(self, tag: str, values: np.ndarray, step: int):
        self.writer.add_histogram(tag, values, step)

    def log_image(self, tag: str, image, step: int, dataformats="HWC"):
        image = np.asarray(image)
        self.writer.add_image(tag, image, step, dataformats=dataformats)
        safe_tag = tag.replace("/", "_").replace(" ", "_")
        filename = f"{safe_tag}_step{step}.png"
        filepath = os.path.join(self.log_dir, filename)
        if image.dtype in [np.float32, np.float64]:
            img_to_save = np.clip(image * 255, 0, 255).astype(np.uint8)
        else:
            img_to_save = image
        imageio.imwrite(filepath, img_to_save)

    def log_text(self, tag: str, text: str, step: int = 0):
        self.writer.add_text(tag, text, step)

    def log_figure(self, tag: str, figure, step: int):
        self.writer.add_figure(tag, figure, step)

    def close(self):
        self.writer.close()


class AimLogger(BaseLogger):
    def __init__(self, log_dir_base: str, config: dict):
        super().__init__(log_dir_base, config)
        self.aim_run = None
        try:
            self.aim_run = aim.Run(experiment=log_dir_base)
            print(f"Aim logging enabled. Repo: {self.aim_run.repo.path}")
            print(f"Aim run hash: {self.aim_run.hash}")
        except Exception as e:
            print(f"Could not initialize Aim, disabling Aim logging. Error: {e}")
            self.aim_run = None
        self._log_hyperparameters()
        self._log_git_info()

    def _log_hyperparameters(self):
        super()._log_hyperparameters()
        if self.aim_run:
            hparams = convert_to_serializable_dict(self.config)
            self.aim_run["hparams"] = hparams

    def _log_git_info(self):
        if not self.aim_run:
            return
        try:
            commit_hash = (
                subprocess.check_output(["git", "rev-parse", "HEAD"]).strip().decode("utf-8")
            )
            self.aim_run["git_commit"] = commit_hash
        except (subprocess.CalledProcessError, FileNotFoundError):
            self.aim_run["git_commit"] = "N/A"

    def log_scalar(self, tag: str, value: float, step: int):
        if self.aim_run:
            self.aim_run.track(float(value), name=tag, step=step)

    def log_histogram(self, tag: str, values: np.ndarray, step: int):
        if self.aim_run:
            try:
                self.aim_run.track(aim.Distribution(values), name=tag, step=step)
            except ValueError as e:
                print(
                    f"Warning: Could not log histogram '{tag}' to Aim at step {step}. "
                    f"This is likely due to all values in the histogram being the same. Error: {e}"
                )

    def log_image(self, tag: str, image, step: int, dataformats="HWC"):
        if self.aim_run:
            image = np.asarray(image)
            if dataformats == "HWC":
                aim_image = image
            else:  # CHW
                aim_image = np.transpose(image, (1, 2, 0))
            self.aim_run.track(aim.Image(aim_image), name=tag, step=step)

        safe_tag = tag.replace("/", "_").replace(" ", "_")
        filename = f"{safe_tag}_step{step}.png"
        filepath = os.path.join(self.log_dir, filename)
        if image.dtype in [np.float32, np.float64]:
            img_to_save = np.clip(image * 255, 0, 255).astype(np.uint8)
        else:
            img_to_save = image
        imageio.imwrite(filepath, img_to_save)

    def log_text(self, tag: str, text: str, step: int = 0):
        if self.aim_run:
            self.aim_run.track(aim.Text(text), name=tag, step=step)

    def log_figure(self, tag: str, figure, step: int):
        if self.aim_run:
            figure.canvas.draw()
            buf = figure.canvas.buffer_rgba()
            width, height = figure.canvas.get_width_height()
            image_from_plot = np.frombuffer(buf, dtype=np.uint8).reshape((height, width, 4))[
                ..., :3
            ]
            self.aim_run.track(aim.Image(image_from_plot), name=tag, step=step)

    def close(self):
        if self.aim_run:
            self.aim_run.close()


class NoOpLogger(BaseLogger):
    def __init__(self):
        print("Logging is disabled.")

    def _create_log_dir(self, log_dir_base: str) -> str:
        # This method is part of the abstract base class but won't be called.
        return "runs/no_op_log"

    def _log_hyperparameters(self):
        pass

    def _log_git_info(self):
        pass

    def log_scalar(self, tag: str, value: float, step: int):
        pass

    def log_histogram(self, tag: str, values: np.ndarray, step: int):
        pass

    def log_image(self, tag: str, image, step: int, dataformats="HWC"):
        pass

    def log_text(self, tag: str, text: str, step: int = 0):
        pass

    def log_figure(self, tag: str, figure, step: int):
        pass

    def close(self):
        pass


def create_logger(logger_type: str, log_dir_base: str, config: dict) -> BaseLogger:
    if logger_type == "aim":
        return AimLogger(log_dir_base, config)
    elif logger_type == "tensorboard":
        return TensorboardLogger(log_dir_base, config)
    elif logger_type == "none":
        return NoOpLogger()
    else:
        raise ValueError(f"Unknown logger type: {logger_type}")
