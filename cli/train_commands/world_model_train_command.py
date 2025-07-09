import chex
import click
import flax.linen as nn
import jax
import jax.numpy as jnp

from config.pydantic_models import WMTrainOptions
from helpers.config_printer import print_config
from helpers.logger import TensorboardLogger
from helpers.rich_progress import trange
from neural_util.optimizer import setup_optimizer
from neural_util.util import round_through_gradient
from world_model_puzzle import WorldModelPuzzleBase
from world_model_puzzle.world_model_train import (
    world_model_eval_builder,
    world_model_train_builder,
)

from ..options import wm_get_ds_options, wm_get_world_model_options, wm_train_options


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
    wm_train_options: WMTrainOptions,
    **kwargs,
):
    config = {
        "world_model_name": world_model_name,
        "world_model": world_model.__class__.__name__,
        "wm_train_options": wm_train_options.dict(),
        "dataset_shapes": {
            "datas": str(datas.shape),
            "next_datas": str(next_datas.shape),
            "actions": str(actions.shape),
            "eval_trajectory_states": str(eval_trajectory[0].shape),
            "eval_trajectory_actions": str(eval_trajectory[1].shape),
        },
        **kwargs,
    }
    print_config("World Model Training Configuration", config)

    logger = TensorboardLogger(world_model_name, config)
    model: nn.Module = world_model.model

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
        params,
        1,
        wm_train_options.train_epochs,
        dataset_size // wm_train_options.mini_batch_size,
        wm_train_options.optimizer,
        lr_init=1e-2,
    )

    print("initializing train function")
    train_fn = world_model_train_builder(
        wm_train_options.mini_batch_size,
        train_info_fn,
        optimizer,
    )

    print("initializing eval function")
    eval_fn = world_model_eval_builder(
        train_info_fn,
        wm_train_options.mini_batch_size,
    )

    print("initializing key")
    key = jax.random.PRNGKey(0)

    print("training")
    pbar = trange(wm_train_options.train_epochs)
    eval_data = (eval_trajectory[0][0], eval_trajectory[0][1], eval_trajectory[1][0])
    logger.log_image("Current/Ground Truth", eval_data[0], 0, dataformats="HWC")
    logger.log_image("Next/Ground Truth", eval_data[1], 0, dataformats="HWC")
    eval_accuracy = eval_fn(params, eval_trajectory)
    logger.log_scalar("Metrics/Eval Accuracy", eval_accuracy, 0)

    for epoch in pbar:
        key, subkey = jax.random.split(key)
        params, opt_state, loss, AE_loss, WM_loss, accuracy = train_fn(
            subkey, (datas, next_datas, actions), params, opt_state, epoch
        )
        lr = opt_state.hyperparams["learning_rate"]
        pbar.set_description(
            desc="Training",
            desc_dict={
                "lr": lr,
                "Loss": float(loss),
                "AE Loss": float(AE_loss),
                "WM Loss": float(WM_loss),
                "Accuracy": float(accuracy),
                "Eval Accuracy": float(eval_accuracy),
            },
        )
        if epoch % 10 == 0:
            logger.log_scalar("Metrics/Learning Rate", lr, epoch)
            logger.log_scalar("Losses/Loss", loss, epoch)
            logger.log_scalar("Losses/AE Loss", AE_loss, epoch)
            logger.log_scalar("Losses/WM Loss", WM_loss, epoch)
            logger.log_scalar("Metrics/Accuracy", accuracy, epoch)

        if epoch % 100 == 0:
            eval_accuracy = eval_fn(params, eval_trajectory)
            logger.log_scalar("Metrics/Eval Accuracy", eval_accuracy, epoch)

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
            logger.log_image("Current/Decoded", decoded[0], epoch, dataformats="HWC")

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
            logger.log_image("Next/Decoded", next_decoded[0], epoch, dataformats="HWC")

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
            logger.log_image("Next/Decoded Pred", next_decoded_pred[0], epoch, dataformats="HWC")

            world_model.params = params
            world_model.save_model()

    logger.close()

    world_model.params = params
    world_model.save_model()
