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
from puzzle.world_model.world_model_train import (
    world_model_eval_builder,
    world_model_train_builder,
)

from .world_model_train_option import (
    get_ds_options,
    get_world_model_options,
    train_options,
)

PyTree = Any


def setup_logging(world_model_name: str) -> tensorboardX.SummaryWriter:
    log_dir = f"runs/{world_model_name}_{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    return tensorboardX.SummaryWriter(log_dir)


def setup_optimizer(params: PyTree, steps: int, one_iter_size: int) -> optax.OptState:
    # Add warmup to the learning rate schedule
    warmup_steps = 10 * one_iter_size

    # Create a warmup schedule that linearly increases from 0 to init_value
    warmup_schedule = optax.linear_schedule(
        init_value=0.0, end_value=1e-3, transition_steps=warmup_steps
    )

    # Create the main decay schedule
    decay_schedule = optax.polynomial_schedule(
        init_value=1e-3,
        end_value=1e-4,
        power=1.0,
        transition_steps=steps * one_iter_size - warmup_steps,
    )

    # Combine the schedules
    lr_schedule = optax.join_schedules(
        schedules=[warmup_schedule, decay_schedule], boundaries=[warmup_steps]
    )

    def adam(learning_rate):
        mask = {"params": True, "batch_stats": False}
        return optax.chain(
            optax.scale_by_adam(),
            optax.add_decayed_weights(1e-5, mask=mask),
            optax.scale_by_learning_rate(learning_rate),
        )

    optimizer = optax.inject_hyperparams(adam)(lr_schedule)
    return optimizer, optimizer.init(params)


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
            method=model.train_info,
            mutable=["batch_stats"],
        )

    params = world_model.params

    dataset_size = actions.shape[0]
    print("initializing optimizer")
    optimizer, opt_state = setup_optimizer(params, train_epochs, dataset_size // mini_batch_size)

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
            f"lr: {lr:.4f}, Loss: {loss:.4f},"
            f"AE Loss: {AE_loss:.4f}, WM Loss: {WM_loss:.4f},"
            f"Accuracy: {accuracy:.4f}, Eval Accuracy: {eval_accuracy:.4f}"
        )
        if epoch % 10 == 0:
            writer.add_scalar("Metrics/Learning Rate", lr, epoch)
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
            world_model.save_model(f"puzzle/world_model/model/params/{world_model_name}.pkl")

    writer.close()

    world_model.params = params
    world_model.save_model(f"puzzle/world_model/model/params/{world_model_name}.pkl")
