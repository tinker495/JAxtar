from datetime import datetime
from typing import Any

import chex
import click
import flax.linen as nn
import jax
import jax.numpy as jnp
import optax
import tensorboardX
from tqdm import trange

from puzzle.gray_world_model.gray_world_model_train import (
    gray_world_model_eval_builder,
    gray_world_model_train_builder,
)
from puzzle.world_model.util import round_through_gradient
from puzzle.world_model.world_model_puzzle_base import WorldModelPuzzleBase

from .world_model_train_option import (
    get_ds_options,
    get_gray_world_model_options,
    train_options,
)

PyTree = Any


def setup_logging(world_model_name: str) -> tensorboardX.SummaryWriter:
    log_dir = f"runs/gray_world_model_{world_model_name}_{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    return tensorboardX.SummaryWriter(log_dir)


def setup_optimizer(params: PyTree) -> optax.OptState:
    optimizer = optax.chain(
        optax.clip_by_global_norm(10.0),  # Clip gradients to a maximum global norm of 1.0
        optax.adamw(1e-3, nesterov=True, weight_decay=1e-4),
    )
    return optimizer, optimizer.init(params)


@click.command()
@get_ds_options
@get_gray_world_model_options
@train_options
def gray_world_model_train(
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

    def train_info_fn(params, data, next_data, training):
        return model.apply(
            params,
            data,
            next_data,
            training=training,
            method=model.train_info,
            mutable=["batch_stats"],
        )

    params = world_model.params

    print("initializing optimizer")
    optimizer, opt_state = setup_optimizer(params)

    print("initializing train function")
    train_fn = gray_world_model_train_builder(
        mini_batch_size,
        train_info_fn,
        optimizer,
    )

    print("initializing eval function")
    eval_fn = gray_world_model_eval_builder(
        train_info_fn,
        mini_batch_size,
    )

    print("initializing key")
    key = jax.random.PRNGKey(0)

    print("training")
    pbar = trange(train_epochs)
    eval_data = (eval_trajectory[0][0], eval_trajectory[0][1], eval_trajectory[1][0])
    writer.add_image("Current/Ground Truth", eval_data[0], 0, dataformats="HWC")
    writer.add_image("Next/Ground Truth", eval_data[1], 0, dataformats="HWC")
    eval_accuracy = eval_fn(params, eval_trajectory)
    writer.add_scalar("Metrics/Eval Accuracy", eval_accuracy, 0)

    for epoch in pbar:
        key, subkey = jax.random.split(key)
        params, opt_state, loss, AE_loss, WM_loss, accuracy = train_fn(
            subkey, (datas, next_datas, actions), params, opt_state, epoch
        )
        pbar.set_description(
            f"Loss: {loss:.4f},"
            f"AE Loss: {AE_loss:.4f}, WM Loss: {WM_loss:.4f},"
            f"Accuracy: {accuracy:.4f}, Eval Accuracy: {eval_accuracy:.4f}"
        )
        if epoch % 10 == 0:
            writer.add_scalar("Losses/Loss", loss, epoch)
            writer.add_scalar("Losses/AE Loss", AE_loss, epoch)
            writer.add_scalar("Losses/WM Loss", WM_loss, epoch)
            writer.add_scalar("Metrics/Accuracy", accuracy, epoch)

            eval_accuracy = eval_fn(params, eval_trajectory)
            writer.add_scalar("Metrics/Eval Accuracy", eval_accuracy, epoch)

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

            flipped = model.apply(params, rounded_latent, training=False, method=model.flipped)
            action = jnp.reshape(
                eval_data[2], (-1,) + (1,) * (flipped.ndim - 1)
            )  # [batch_size, 1, ...]
            next_flipped = jnp.take_along_axis(flipped, action, axis=1).squeeze(
                axis=1
            )  # [batch_size, latent_size]
            rounded_flipped = jnp.round(next_flipped)
            pred_latent = jnp.logical_xor(rounded_latent, rounded_flipped).astype(jnp.float32)
            next_decoded_pred = jnp.clip(
                model.apply(params, pred_latent, training=False, method=model.decode) * 255.0 / 2.0
                + 128.0,
                0,
                255,
            ).astype(jnp.uint8)
            writer.add_image("Next/Decoded Pred", next_decoded_pred[0], epoch, dataformats="HWC")

            world_model.params = params
            world_model.save_model(f"puzzle/gray_world_model/model/params/{world_model_name}.pkl")

    writer.close()
