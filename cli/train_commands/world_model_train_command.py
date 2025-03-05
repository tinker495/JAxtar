from datetime import datetime
from typing import Any

import chex
import click
import flax.linen as nn
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import optax
import tensorboardX
from tqdm import trange

from puzzle.world_model.util import round_through_gradient
from puzzle.world_model.world_model_puzzle_base import WorldModelPuzzleBase
from puzzle.world_model.world_model_train import (
    world_model_eval_builder,
    world_model_train_builder,
)

from .world_model_train_option import (
    get_ds_options,
    get_world_model_options,
    train_options,
)

USE_UMAP = False

PyTree = Any

if USE_UMAP:
    import umap

    reducer = umap.UMAP()
else:
    import sklearn.manifold

    reducer = sklearn.manifold.TSNE(n_components=2, random_state=0)


def setup_logging(world_model_name: str) -> tensorboardX.SummaryWriter:
    log_dir = f"runs/world_model_{world_model_name}_{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    return tensorboardX.SummaryWriter(log_dir)


def setup_optimizer(params: PyTree) -> optax.OptState:
    optimizer = optax.chain(
        optax.clip_by_global_norm(10.0), optax.adamw(1e-3, nesterov=True, weight_decay=1e-6)
    )
    return optimizer, optimizer.init(params)


def visualize_latents(latents, epoch, prefix="Latents"):
    # Convert to numpy for sklearn

    latents_np = np.array(latents)

    # Reshape if needed to 2D array [n_samples, n_features]
    if len(latents_np.shape) > 2:
        latents_np = latents_np.reshape(latents_np.shape[0], -1)

    latents_var = np.var(latents_np, axis=0)
    latents_var = np.mean(latents_var)

    # Calculate the angle variance of latent vectors
    # First normalize the latent vectors
    latents_normalized = latents_np / (np.linalg.norm(latents_np, axis=1, keepdims=True) + 1e-8)

    # Calculate the dot products between consecutive latent vectors
    dot_products = np.sum(latents_normalized[:-1] * latents_normalized[1:], axis=1)

    # Clip dot products to valid range for arccos
    dot_products = np.clip(dot_products, -1.0, 1.0)

    # Calculate angles in radians and convert to degrees
    angles = np.arccos(dot_products) * (180.0 / np.pi)

    # Calculate angle variance
    angle_variance = np.var(angles)

    # Apply dimensionality reduction
    latents_2d = reducer.fit_transform(latents_np)

    # Create plot
    fig, ax = plt.subplots(figsize=(10, 8))

    # Color points according to their original ordering
    num_points = latents_np.shape[0]
    point_order = np.arange(num_points)

    # Use a colormap to represent ordering
    scatter = ax.scatter(
        latents_2d[:, 0], latents_2d[:, 1], c=point_order, cmap="viridis", alpha=0.7, s=30
    )

    # Add colorbar to show the relationship between colors and ordering
    cbar = fig.colorbar(scatter, ax=ax)
    cbar.set_label("Point order in original sequence")

    ax.set_title(
        f"{prefix}(Epoch: {epoch}) UMAP\n"
        f"Variance: {latents_var:.4f}, "
        f"Angle Variance: {angle_variance:.4f}"
    )
    ax.set_xlabel("Dimension 1")
    ax.set_ylabel("Dimension 2")

    # Convert plot to image for tensorboard
    fig.canvas.draw()
    width, height = fig.canvas.get_width_height()
    image = np.frombuffer(fig.canvas.tostring_argb(), dtype=np.uint8)
    image = image.reshape((height, width, 4))[:, :, 1:]  # ARGB to RGB
    plt.close(fig)

    return image


@click.command()
@get_ds_options
@get_world_model_options
@train_options
def train(
    world_model_name: str,
    datas: chex.Array,
    next_datas: chex.Array,
    actions: chex.Array,
    eval_trajectory: tuple[chex.Array, chex.Array],
    world_model: WorldModelPuzzleBase,
    train_epochs: int,
    mini_batch_size: int,
    **kwargs,
):

    writer = setup_logging(world_model_name)
    model: nn.Model = world_model.model

    def train_info_fn(params, data, next_data, action, training):
        return model.apply(
            params,
            data,
            next_data,
            action,
            training=training,
            method=model.world_model_train_info,
            mutable=["batch_stats"],
        )

    def get_projections_fn(params, latent, training=False):
        return model.apply(
            params,
            latent,
            training=training,
            method=model.get_projections,
        )

    params = world_model.params
    target_params = world_model.params

    print("initializing optimizer")
    optimizer, opt_state = setup_optimizer(params)

    print("initializing train function")
    train_fn = world_model_train_builder(
        mini_batch_size,
        train_info_fn,
        get_projections_fn,
        optimizer,
    )

    print("initializing eval function")
    eval_fn = world_model_eval_builder(
        train_info_fn,
        get_projections_fn,
        mini_batch_size,
    )

    print("initializing key")
    key = jax.random.PRNGKey(0)

    print("training")
    pbar = trange(train_epochs)
    eval_data = (eval_trajectory[0][0], eval_trajectory[0][1], eval_trajectory[1][0])
    writer.add_image("Current/Ground Truth", eval_data[0], 0, dataformats="HWC")
    writer.add_image("Next/Ground Truth", eval_data[1], 0, dataformats="HWC")
    eval_accuracy, eval_forward_projected_latents, eval_backward_projected_latents = eval_fn(
        params, eval_trajectory
    )
    writer.add_scalar("Metrics/Eval Accuracy", eval_accuracy, 0)

    forward_tsne_img = visualize_latents(
        eval_forward_projected_latents, 0, "Forward Projected Latents"
    )
    writer.add_image("Latent/Forward Projected Latents", forward_tsne_img, 0, dataformats="HWC")

    backward_tsne_img = visualize_latents(
        eval_backward_projected_latents, 0, "Backward Projected Latents"
    )
    writer.add_image("Latent/Backward Projected Latents", backward_tsne_img, 0, dataformats="HWC")

    for epoch in pbar:
        key, subkey = jax.random.split(key)
        (
            params,
            target_params,
            opt_state,
            loss,
            AE_loss,
            world_model_loss,
            total_projection_distance_loss,
            orthonormality_regularization_loss,
            accuracy,
            target_Qs,
            current_Qs,
        ) = train_fn(subkey, (datas, next_datas, actions), params, target_params, opt_state, epoch)
        mean_target_Qs = jnp.mean(target_Qs)
        mean_current_Qs = jnp.mean(current_Qs)
        pbar.set_description(
            f"Loss: {loss:.4f}, "
            f"AE Loss: {AE_loss:.4f}, "
            f"WM Loss: {world_model_loss:.4f}, "
            f"PD Loss: {total_projection_distance_loss:.4f}, "
            f"OR Loss: {orthonormality_regularization_loss:.4f}, "
            f"Acc: {accuracy*100:3.1f}, "
            f"Eval Acc: {eval_accuracy*100:3.1f}, "
            f"TQs: {mean_target_Qs:.4f}, "
            f"CQs: {mean_current_Qs:.4f}"
        )
        if epoch % 10 == 0:
            writer.add_scalar("Losses/Loss", loss, epoch)
            writer.add_scalar("Losses/AE Loss", AE_loss, epoch)
            writer.add_scalar("Losses/World Model Loss", world_model_loss, epoch)
            writer.add_scalar(
                "Losses/Projection Distance Loss", total_projection_distance_loss, epoch
            )
            writer.add_scalar(
                "Losses/Orthonormality Regularization Loss",
                orthonormality_regularization_loss,
                epoch,
            )
            writer.add_scalar("Metrics/Accuracy", accuracy, epoch)
            writer.add_scalar("Metrics/Mean Target Qs", mean_target_Qs, epoch)
            writer.add_scalar("Metrics/Mean Current Qs", mean_current_Qs, epoch)

            writer.add_histogram("Target Qs", target_Qs[:1000], epoch)
            writer.add_histogram("Current Qs", current_Qs[:1000], epoch)

        if epoch % 100 == 0:
            (
                eval_accuracy,
                eval_forward_projected_latents,
                eval_backward_projected_latents,
            ) = eval_fn(params, eval_trajectory)
            writer.add_scalar("Metrics/Eval Accuracy", eval_accuracy, epoch)

            forward_tsne_img = visualize_latents(
                eval_forward_projected_latents, epoch, "Forward Projected Latents"
            )
            writer.add_image(
                "Latent/Forward Projected Latents", forward_tsne_img, epoch, dataformats="HWC"
            )

            backward_tsne_img = visualize_latents(
                eval_backward_projected_latents, epoch, "Backward Projected Latents"
            )
            writer.add_image(
                "Latent/Backward Projected Latents", backward_tsne_img, epoch, dataformats="HWC"
            )

            data = jnp.expand_dims(eval_data[0], axis=0)
            latent = model.apply(params, data, training=False, method=model.encode)
            rounded_latent = round_through_gradient(latent)
            decoded = jnp.clip(
                model.apply(params, rounded_latent, training=False, method=model.decode)
                * 255.0
                / 2.0
                + 128.0,
                0,
                255,
            ).astype(jnp.uint8)
            writer.add_image("Current/Decoded", decoded[0], epoch, dataformats="HWC")

            next_data = jnp.expand_dims(eval_data[1], axis=0)
            next_latent = model.apply(params, next_data, training=False, method=model.encode)
            next_rounded_latent = round_through_gradient(next_latent)
            next_decoded = jnp.clip(
                model.apply(params, next_rounded_latent, training=False, method=model.decode)
                * 255.0
                / 2.0
                + 128.0,
                0,
                255,
            ).astype(jnp.uint8)
            writer.add_image("Next/Decoded", next_decoded[0], epoch, dataformats="HWC")

            next_latent_pred = model.apply(
                params, rounded_latent, training=False, method=model.transition
            )
            action = jnp.reshape(
                eval_data[2], (-1,) + (1,) * (next_latent_pred.ndim - 1)
            )  # [batch_size, 1, ...]
            next_latent_pred = jnp.take_along_axis(next_latent_pred, action, axis=1).squeeze(
                axis=1
            )  # [batch_size, ...]
            next_latent_pred = round_through_gradient(next_latent_pred)
            next_decoded_pred = jnp.clip(
                model.apply(params, next_latent_pred, training=False, method=model.decode)
                * 255.0
                / 2.0
                + 128.0,
                0,
                255,
            ).astype(jnp.uint8)
            writer.add_image("Next/Decoded Pred", next_decoded_pred[0], epoch, dataformats="HWC")

            world_model.params = params
            world_model.save_model(f"puzzle/world_model/model/params/{world_model_name}_repr.pkl")

    writer.close()
