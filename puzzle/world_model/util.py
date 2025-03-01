import os
from functools import lru_cache, wraps

import chex
import cv2
import huggingface_hub
import jax
import jax.numpy as jnp
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from sklearn.manifold import TSNE

matplotlib.use("Agg")  # Use the 'Agg' backend which doesn't require a display


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
    return x + jax.lax.stop_gradient(jnp.round(x).astype(jnp.float32) - x)


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


def np_cache(function):
    @lru_cache()
    def cached_wrapper(hashable_array):
        array = np.array(hashable_array)
        return function(array)

    @wraps(function)
    def wrapper(array):
        # Convert array to a hashable type (tuple of tuples)
        array_tuple = tuple(map(tuple, array)) if array.ndim > 1 else tuple(array)
        return cached_wrapper(array_tuple)

    # copy lru_cache attributes over too
    wrapper.cache_info = cached_wrapper.cache_info
    wrapper.cache_clear = cached_wrapper.cache_clear

    return wrapper


@np_cache
def tsne_fit_transform(latents):
    # Create a t-SNE embedding of the projected latents
    tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(latents) - 1))
    tsne_results = tsne.fit_transform(latents)
    return tsne_results


def latents_to_tsne_img(
    latents: np.ndarray, idx: int = None, figsize=(4, 4), dpi=100, img_width: int = 32
) -> np.ndarray:
    """
    Convert latent vectors to a t-SNE visualization image.

    Args:
        latents: Array of latent vectors to visualize
        idx: Index of the current point to highlight (if None, no point is highlighted)
        figsize: Size of the matplotlib figure
        dpi: DPI of the matplotlib figure

    Returns:
        np.ndarray: BGR image of the t-SNE visualization
    """

    tsne_results = tsne_fit_transform(latents)
    # Create a matplotlib figure for the t-SNE visualization
    fig = plt.figure(figsize=figsize, dpi=dpi)
    ax = fig.add_subplot(111)

    # Plot all points
    ax.scatter(tsne_results[:, 0], tsne_results[:, 1], c="lightgray", s=30)

    # Highlight the path with a line
    ax.plot(tsne_results[:, 0], tsne_results[:, 1], "b-", linewidth=1, alpha=0.7)

    # Mark the start point
    ax.scatter(tsne_results[0, 0], tsne_results[0, 1], c="green", s=50, label="Start")

    # Mark the current point
    ax.scatter(tsne_results[idx, 0], tsne_results[idx, 1], c="red", s=50, label="Current")

    # Add a legend
    ax.legend(loc="best", fontsize=8)

    # Remove axes
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title("t-SNE of Latent Space", fontsize=10)

    # Convert the matplotlib figure to an image for visualization
    fig.canvas.draw()
    width, height = fig.canvas.get_width_height()
    tsne_img = np.frombuffer(fig.canvas.tostring_argb(), dtype=np.uint8)
    tsne_img = tsne_img.reshape((height, width, 4))[:, :, 1:]  # ARGB to RGB
    tsne_img = cv2.cvtColor(tsne_img, cv2.COLOR_RGB2BGR)

    # Resize the t-SNE image to match the width of the main image
    tsne_img = cv2.resize(
        tsne_img, (img_width, int(width * img_width / width)), interpolation=cv2.INTER_AREA
    )
    return tsne_img
