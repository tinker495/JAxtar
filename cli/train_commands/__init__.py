"""Lightweight tyro training entry points; load training backends on execution."""

from dataclasses import asdict

import tyro

from ..options import (
    HeuristicTrainArgs,
    QFunctionTrainArgs,
    resolve_puzzle,
    resolve_train,
    resolve_train_heuristic,
    resolve_train_qfunction,
)
from ..runtime import run_cli

distance_train_app = tyro.extras.SubcommandApp()


@distance_train_app.command(name="heuristic")
def _heuristic(options: HeuristicTrainArgs):
    kwargs = resolve_puzzle(asdict(options), default_hard=True, use_seeds_flag=False)
    kwargs = resolve_train(kwargs, preset_category="heuristic_train")
    kwargs = resolve_train_heuristic(kwargs)
    from .dist_train_command import run_heuristic_training

    return run_heuristic_training(**kwargs)


@distance_train_app.command(name="qfunction")
def _qfunction(options: QFunctionTrainArgs):
    kwargs = resolve_puzzle(asdict(options), default_hard=True, use_seeds_flag=False)
    kwargs = resolve_train(kwargs, preset_category="qfunction_train")
    kwargs = resolve_train_qfunction(kwargs)
    from .dist_train_command import run_qfunction_training

    return run_qfunction_training(**kwargs)


def heuristic_train_command(args=None):
    return run_cli(lambda **kwargs: tyro.cli(_heuristic, **kwargs), args)


def qfunction_train_command(args=None):
    return run_cli(lambda **kwargs: tyro.cli(_qfunction, **kwargs), args)


def distance_train(args=None):
    return run_cli(
        distance_train_app.cli,
        args,
        prog="distance-train",
        description="Train neural heuristic and Q-function distance estimators.",
    )
