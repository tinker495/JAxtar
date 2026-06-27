"""Heuristic training builder."""

from __future__ import annotations

from typing import Any, Callable

import optax

from neural_util.basemodel import DistanceHLGModel, DistanceModel
from train_util.distance_train_builder import distance_train_builder


def heuristic_train_builder(
    minibatch_size: int,
    heuristic_model: DistanceModel | DistanceHLGModel,
    optimizer: optax.GradientTransformation,
    preproc_fn: Callable,
    n_devices: int = 1,
    loss_type: str = "mse",
    loss_args: dict[str, Any] | None = None,
    replay_ratio: int = 1,
    use_soft_update: bool = False,
    update_interval: int = 100,
    soft_update_tau: float = 0.005,
    enable_jit_hard_update: bool = True,
):
    return distance_train_builder(
        minibatch_size=minibatch_size,
        model=heuristic_model,
        optimizer=optimizer,
        preproc_fn=preproc_fn,
        target_keys=("target_heuristic",),
        n_devices=n_devices,
        loss_type=loss_type,
        loss_args=loss_args,
        replay_ratio=replay_ratio,
        use_soft_update=use_soft_update,
        update_interval=update_interval,
        soft_update_tau=soft_update_tau,
        enable_jit_hard_update=enable_jit_hard_update,
    )
