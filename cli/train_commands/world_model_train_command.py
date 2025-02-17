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

from puzzle.world_model.util import round_through_gradient
from puzzle.world_model.world_model_puzzle_base import WorldModelPuzzleBase
from puzzle.world_model.world_model_train import world_model_train_builder

from .world_model_train_option import (
    get_ds_options,
    get_world_model_options,
    train_options,
)

PyTree = Any


def setup_logging(dataset: str) -> tensorboardX.SummaryWriter:
    log_dir = f"runs/{dataset}_{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    return tensorboardX.SummaryWriter(log_dir)


def setup_optimizer(params: PyTree) -> optax.OptState:
    optimizer = optax.chain(
        optax.clip_by_global_norm(10.0),  # Clip gradients to a maximum global norm of 1.0
        optax.adam(1e-3, nesterov=True),
    )
    return optimizer, optimizer.init(params)


@click.command()
@get_ds_options
@get_world_model_options
@train_options
def train(
    dataset: str,
    datas: chex.Array,
    next_datas: chex.Array,
    actions: chex.Array,
    world_model: WorldModelPuzzleBase,
    train_steps: int,
    mini_batch_size: int,
    **kwargs,
):

    writer = setup_logging(dataset)
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
    train_fn = world_model_train_builder(
        mini_batch_size,
        train_info_fn,
        optimizer,
    )

    print("initializing key")
    key = jax.random.PRNGKey(0)

    print("training")
    pbar = trange(train_steps)
    eval_data = (datas[0], next_datas[0], actions[0])
    writer.add_image("Current/Ground Truth", eval_data[0], 0, dataformats="HWC")
    writer.add_image("Next/Ground Truth", eval_data[1], 0, dataformats="HWC")

    for i in pbar:
        key, subkey = jax.random.split(key)
        params, opt_state, loss, AE_loss, WM_loss, accuracy = train_fn(
            subkey, (datas, next_datas, actions), params, opt_state
        )
        pbar.set_description(
            f"Loss: {loss:.4f}, AE Loss: {AE_loss:.4f}, WM Loss: {WM_loss:.4f}, Accuracy: {accuracy:.4f}"
        )
        if i % 10 == 0:
            writer.add_scalar("Losses/Loss", loss, i)
            writer.add_scalar("Losses/AE Loss", AE_loss, i)
            writer.add_scalar("Losses/WM Loss", WM_loss, i)
            writer.add_scalar("Metrics/Accuracy", accuracy, i)

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
            writer.add_image("Current/Decoded", decoded[0], i, dataformats="HWC")

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
            writer.add_image("Next/Decoded", next_decoded[0], i, dataformats="HWC")

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
            writer.add_image("Next/Decoded Pred", next_decoded_pred[0], i, dataformats="HWC")

            world_model.params = params
            world_model.save_model(f"puzzle/world_model/model/params/{dataset}.pkl")

    writer.close()
