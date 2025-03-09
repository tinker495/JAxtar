import os

import chex
import huggingface_hub
import jax
import jax.numpy as jnp
import numpy as np


def is_dataset_downloaded():
    return os.path.exists("puzzle/world_model/data")


def download_dataset():
    huggingface_hub.snapshot_download(
        repo_id="Tinker/puzzle_world_model_ds",
        repo_type="dataset",
        local_dir="puzzle/world_model/",
    )


def is_model_downloaded(filename: str):
    return os.path.exists(filename)


def download_model(filename: str):
    huggingface_hub.hf_hub_download(
        repo_id="Tinker/JAxtar_models",
        repo_type="model",
        filename=filename,
        local_dir="",
    )


def round_through_gradient(x: chex.Array) -> chex.Array:
    # x is a sigmoided value in the range [0, 1]. Use a straight-through estimator:
    # the forward pass returns jnp.round(x) while the gradient flows as if it were the identity.
    return x + jax.lax.stop_gradient(jnp.where(x > 0.5, 1.0, 0.0).astype(jnp.float32) - x)


def img_to_colored_str(img: np.ndarray) -> str:
    """
    Convert a numpy array to an ascii string.
    img size = (32, 32, 3)
    """
    if img.dtype != np.uint8:
        img = (img * 255).astype(np.uint8)
    ascii_art_lines = []
    for row in img:
        line = ""
        for pixel in row:
            r, g, b = int(pixel[0]), int(pixel[1]), int(pixel[2])
            char = "██"
            # Append the character with ANSI escape codes to reflect its original color
            line += f"\x1b[38;2;{r};{g};{b}m{char}\x1b[0m"
        ascii_art_lines.append(line)
    return "\n".join(ascii_art_lines)
