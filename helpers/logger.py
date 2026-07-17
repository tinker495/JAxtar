import importlib
import os
import shutil
import subprocess
from abc import ABC, abstractmethod
from datetime import datetime
from typing import Optional

import imageio.v2 as imageio
import numpy as np

from helpers.util import convert_to_serializable_dict


def _optional_module(module_name: str, feature: str):
    try:
        return importlib.import_module(module_name)
    except ImportError as exc:
        raise ImportError(
            f"{feature} requires optional logging dependencies. "
            "Install them with `pip install -r requirements_logging.txt`."
        ) from exc


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

    def _get_git_commit_hash(self) -> str:
        try:
            return (
                subprocess.check_output(["git", "rev-parse", "HEAD"], timeout=5)
                .strip()
                .decode("utf-8")
            )
        except (subprocess.CalledProcessError, FileNotFoundError):
            return "N/A"

    def _save_image_local(self, tag: str, image: np.ndarray, step: int):
        safe_tag = tag.replace("/", "_").replace(" ", "_")
        filename = f"{safe_tag}_step{step}.png"
        filepath = os.path.join(self.log_dir, filename)

        if image.dtype in [np.float32, np.float64]:
            img_to_save = np.clip(image * 255, 0, 255).astype(np.uint8)
        else:
            img_to_save = image
        imageio.imwrite(filepath, img_to_save)

    def _save_artifact_local(
        self,
        artifact_path: str,
        artifact_name: str = None,
        artifact_type: str = "model",
    ) -> Optional[str]:
        if not os.path.exists(artifact_path):
            print(f"Warning: Artifact path {artifact_path} does not exist")
            return None

        if artifact_name is None:
            artifact_name = os.path.basename(artifact_path)

        # Create artifacts subdirectory
        artifacts_dir = os.path.join(self.log_dir, "artifacts", artifact_type)
        os.makedirs(artifacts_dir, exist_ok=True)

        # Copy artifact to artifacts directory
        dest_path = os.path.join(artifacts_dir, artifact_name)
        if os.path.isdir(artifact_path):
            shutil.copytree(artifact_path, dest_path, dirs_exist_ok=True)
        else:
            shutil.copy2(artifact_path, dest_path)

        print(f"Artifact saved to: {dest_path}")
        return dest_path

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
    def log_figure(self, tag: str, figure, step: int):
        pass

    @abstractmethod
    def log_artifact(
        self,
        artifact_path: str,
        artifact_name: str = None,
        artifact_type: str = "model",
    ):
        pass

    @abstractmethod
    def close(self):
        pass


class TensorboardLogger(BaseLogger):
    def __init__(self, log_dir_base: str, config: dict):
        super().__init__(log_dir_base, config)
        tensorboardX = _optional_module("tensorboardX", "Tensorboard logging")
        self.writer = tensorboardX.SummaryWriter(self.log_dir)
        print(f"Tensorboard log directory: {self.log_dir}")
        self._log_hyperparameters()
        self._log_git_info()

    def _log_hyperparameters(self):
        super()._log_hyperparameters()
        config_str = "\n".join([f"{key}: {value}" for key, value in self.config.items()])
        self.writer.add_text("Configuration", config_str)

    def _log_git_info(self):
        self.writer.add_text("Git Commit", self._get_git_commit_hash())

    def log_scalar(self, tag: str, value: float, step: int):
        self.writer.add_scalar(tag, value, step)

    def log_histogram(self, tag: str, values: np.ndarray, step: int):
        self.writer.add_histogram(tag, values, step)

    def log_image(self, tag: str, image, step: int, dataformats="HWC"):
        image = np.asarray(image)
        self.writer.add_image(tag, image, step, dataformats=dataformats)
        self._save_image_local(tag, image, step)

    def log_figure(self, tag: str, figure, step: int):
        self.writer.add_figure(tag, figure, step)

    def log_artifact(
        self,
        artifact_path: str,
        artifact_name: str = None,
        artifact_type: str = "model",
    ):
        dest_path = self._save_artifact_local(artifact_path, artifact_name, artifact_type)
        if dest_path:
            artifact_name = artifact_name or os.path.basename(artifact_path)
            artifact_info = f"Artifact: {artifact_name}\nType: {artifact_type}\nPath: {dest_path}"
            self.writer.add_text(f"Artifacts/{artifact_type}/{artifact_name}", artifact_info)

    def close(self):
        self.writer.close()


class AimLogger(BaseLogger):
    def __init__(self, log_dir_base: str, config: dict):
        super().__init__(log_dir_base, config)
        self.aim_run = None
        self._aim = None
        try:
            self._aim = _optional_module("aim", "Aim logging")
            self.aim_run = self._aim.Run(experiment=log_dir_base)
            print(f"Aim logging enabled. Repo: {self.aim_run.repo.path}")
            print(f"Aim run hash: {self.aim_run.hash}")
        except (ImportError, ConnectionError, ValueError, RuntimeError) as e:
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
        if self.aim_run:
            self.aim_run["git_commit"] = self._get_git_commit_hash()

    def log_scalar(self, tag: str, value: float, step: int):
        if self.aim_run:
            self.aim_run.track(float(value), name=tag, step=step)

    def log_histogram(self, tag: str, values: np.ndarray, step: int):
        if self.aim_run:
            try:
                self.aim_run.track(self._aim.Distribution(values), name=tag, step=step)
            except ValueError as e:
                print(
                    f"Warning: Could not log histogram '{tag}' to Aim at step {step}. "
                    f"This is likely due to all values in the histogram being the same. Error: {e}"
                )

    def log_image(self, tag: str, image, step: int, dataformats="HWC"):
        image = np.asarray(image)
        if self.aim_run:
            if dataformats == "HWC":
                aim_image = image
            else:  # CHW
                aim_image = np.transpose(image, (1, 2, 0))
            self.aim_run.track(self._aim.Image(aim_image), name=tag, step=step)

        self._save_image_local(tag, image, step)

    def log_figure(self, tag: str, figure, step: int):
        if self.aim_run:
            figure.canvas.draw()
            buf = figure.canvas.buffer_rgba()
            width, height = figure.canvas.get_width_height()
            image_from_plot = np.frombuffer(buf, dtype=np.uint8).reshape((height, width, 4))[
                ..., :3
            ]
            self.aim_run.track(self._aim.Image(image_from_plot), name=tag, step=step)

    def log_artifact(
        self,
        artifact_path: str,
        artifact_name: str = None,
        artifact_type: str = "model",
    ):
        dest_path = self._save_artifact_local(artifact_path, artifact_name, artifact_type)

        if self.aim_run and dest_path:
            artifact_name = artifact_name or os.path.basename(artifact_path)
            artifact_info = {
                "name": artifact_name,
                "type": artifact_type,
                "path": dest_path,
                "original_path": artifact_path,
                "size": (os.path.getsize(dest_path) if os.path.isfile(dest_path) else "directory"),
            }
            self.aim_run.set(("artifacts", artifact_type, artifact_name), artifact_info)

    def close(self):
        if self.aim_run:
            self.aim_run.close()


class WandbLogger(BaseLogger):
    def __init__(self, log_dir_base: str, config: dict):
        super().__init__(log_dir_base, config)
        self.wandb_run = None
        self._wandb = None
        try:
            self._wandb = _optional_module("wandb", "Wandb logging")
            # Initialize wandb with project name and config
            # Support for run organization through config
            init_kwargs = {
                "project": log_dir_base,
                "config": convert_to_serializable_dict(config),
                "dir": self.log_dir,
                "reinit": "finish_previous",
            }

            # Optional run organization parameters
            if "run_name" in config:
                init_kwargs["name"] = config["run_name"]
            if "wandb_group" in config:
                init_kwargs["group"] = config["wandb_group"]
            if "wandb_job_type" in config:
                init_kwargs["job_type"] = config["wandb_job_type"]

            self.wandb_run = self._wandb.init(**init_kwargs)
            print(f"Wandb logging enabled. Project: {log_dir_base}")
            print(f"Wandb run URL: {self.wandb_run.url}")
        except (ImportError, ConnectionError, ValueError, RuntimeError) as e:
            print(f"Could not initialize Wandb, disabling Wandb logging. Error: {e}")
            self.wandb_run = None
        self._log_hyperparameters()
        self._log_git_info()

    def _log_hyperparameters(self):
        super()._log_hyperparameters()
        # Wandb automatically logs config during init, but we can also update it
        if self.wandb_run:
            self._wandb.config.update(
                convert_to_serializable_dict(self.config), allow_val_change=True
            )

    def _log_git_info(self):
        if self.wandb_run:
            self._wandb.config.update(
                {"git_commit": self._get_git_commit_hash()}, allow_val_change=True
            )

    def log_scalar(self, tag: str, value: float, step: int):
        if self.wandb_run:
            self._wandb.log({tag: float(value)}, step=step)

    def log_histogram(self, tag: str, values: np.ndarray, step: int):
        if self.wandb_run:
            try:
                self._wandb.log({tag: self._wandb.Histogram(values)}, step=step)
            except ValueError as e:
                print(
                    f"Warning: Could not log histogram '{tag}' to Wandb at step {step}. "
                    f"This is likely due to all values in the histogram being the same. Error: {e}"
                )

    def log_image(self, tag: str, image, step: int, dataformats="HWC"):
        image = np.asarray(image)
        if self.wandb_run:
            if dataformats == "CHW":
                # Convert CHW to HWC for wandb
                image = np.transpose(image, (1, 2, 0))
            self._wandb.log({tag: self._wandb.Image(image)}, step=step)

        self._save_image_local(tag, image, step)

    def log_figure(self, tag: str, figure, step: int):
        if self.wandb_run:
            # Wrap figure with wandb.Image for better portability across backends
            self._wandb.log({tag: self._wandb.Image(figure)}, step=step)

    def log_artifact(
        self,
        artifact_path: str,
        artifact_name: str = None,
        artifact_type: str = "model",
        aliases: list = None,
    ):
        """
        Log artifact to Wandb using native artifact support.
        Wandb has excellent native artifact tracking capabilities.
        """
        if not self.wandb_run:
            print("Warning: Wandb run not initialized, cannot log artifact")
            return

        if not os.path.exists(artifact_path):
            print(f"Warning: Artifact path {artifact_path} does not exist")
            return

        if artifact_name is None:
            artifact_name = os.path.basename(artifact_path)

        if aliases is None:
            aliases = ["latest"]

        try:
            artifact_kwargs = {
                "name": artifact_name,
                "type": artifact_type,
                "description": f"Artifact: {artifact_name} of type {artifact_type}",
            }
            artifact = self._wandb.Artifact(**artifact_kwargs)

            # Add file or directory to artifact
            if os.path.isdir(artifact_path):
                artifact.add_dir(artifact_path)
            else:
                artifact.add_file(artifact_path)

            # Log artifact to wandb with explicit run context and aliases
            self.wandb_run.log_artifact(artifact, aliases=aliases)
            print(
                f"Artifact '{artifact_name}' logged to Wandb successfully with aliases: {aliases}"
            )

        except (AttributeError, ValueError, RuntimeError, ConnectionError) as e:
            print(f"Error logging artifact to Wandb: {e}")
            # Fallback: copy to local directory like other loggers
            self._save_artifact_local(artifact_path, artifact_name, artifact_type)

    def close(self):
        if self.wandb_run:
            self._wandb.finish()


class NoOpLogger(BaseLogger):
    def __init__(self, log_dir_base: str, config: dict):
        # Keep the BaseLogger contract (log_dir) so checkpoint saving still works.
        super().__init__(log_dir_base, config)
        print("Logging is disabled.")

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

    def log_figure(self, tag: str, figure, step: int):
        pass

    def log_artifact(
        self,
        artifact_path: str,
        artifact_name: str = None,
        artifact_type: str = "model",
    ):
        pass

    def close(self):
        pass


def create_logger(logger_type: str, log_dir_base: str, config: dict) -> BaseLogger:
    if logger_type == "aim":
        return AimLogger(log_dir_base, config)
    elif logger_type == "tensorboard":
        return TensorboardLogger(log_dir_base, config)
    elif logger_type == "wandb":
        return WandbLogger(log_dir_base, config)
    elif logger_type == "none":
        return NoOpLogger(log_dir_base, config)
    else:
        raise ValueError(f"Unknown logger type: {logger_type}")
