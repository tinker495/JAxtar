from datetime import datetime

import chex
import click
import flax.linen as nn
import jax
import jax.numpy as jnp
import tensorboardX
from tqdm import trange

from neural_util.optimizer import setup_optimizer
from neural_util.util import round_through_gradient
from world_model_puzzle import WorldModelPuzzleBase
from world_model_puzzle.world_model_train import (
    world_model_eval_builder,
    world_model_train_builder,
)

from ..options import wm_get_ds_options, wm_get_world_model_options, wm_train_options


def setup_logging(world_model_name: str) -> tensorboardX.SummaryWriter:
    log_dir = f"runs/{world_model_name}_{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    return tensorboardX.SummaryWriter(log_dir)


@click.command()
@wm_get_ds_options
@wm_get_world_model_options
@wm_train_options
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
            method=model.train_info,
            mutable=["batch_stats"],
        )

    params = world_model.params

    dataset_size = actions.shape[0]
    print("initializing optimizer")
    optimizer, opt_state = setup_optimizer(
        params, 1, train_epochs, dataset_size // mini_batch_size, lr_init=1e-2
    )

    print("initializing train function")
    train_fn = world_model_train_builder(
        mini_batch_size,
        train_info_fn,
        optimizer,
    )

    print("initializing eval function")
    eval_fn = world_model_eval_builder(
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
        lr = opt_state.hyperparams["learning_rate"]
        pbar.set_description(
            f"lr: {lr:.4f}, Loss: {float(loss):.4f},"
            f"AE Loss: {float(AE_loss):.4f}, WM Loss: {float(WM_loss):.4f},"
            f"Accuracy: {float(accuracy):.4f}, Eval Accuracy: {float(eval_accuracy):.4f}"
        )
        if epoch % 10 == 0:
            writer.add_scalar("Metrics/Learning Rate", lr, epoch)
            writer.add_scalar("Losses/Loss", loss, epoch)
            writer.add_scalar("Losses/AE Loss", AE_loss, epoch)
            writer.add_scalar("Losses/WM Loss", WM_loss, epoch)
            writer.add_scalar("Metrics/Accuracy", accuracy, epoch)

        if epoch % 100 == 0:
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
            world_model.save_model()

    writer.close()

    world_model.params = params
    world_model.save_model()
