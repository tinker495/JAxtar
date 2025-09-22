from __future__ import annotations

import json
from typing import Any, Dict, Mapping, Tuple

from omegaconf import DictConfig, OmegaConf
from puxle import Puzzle

from config import puzzle_bundles
from config.pydantic_models import (
    DistQFunctionOptions,
    DistTrainOptions,
    EvalOptions,
    NeuralCallableConfig,
    PuzzleBundle,
    PuzzleConfig,
    PuzzleOptions,
    WorldModelPuzzleConfig,
)
from heuristic.neuralheuristic.neuralheuristic_base import NeuralHeuristicBase
from qfunction.neuralq.neuralq_base import NeuralQFunctionBase


def _to_plain_dict(config: Mapping[str, Any] | DictConfig | None) -> Dict[str, Any]:
    if config is None:
        return {}
    if isinstance(config, DictConfig):
        return OmegaConf.to_container(config, resolve=True)  # type: ignore[return-value]
    return dict(config)


def build_puzzle(config: Mapping[str, Any] | DictConfig | None) -> Tuple[
    Puzzle,
    PuzzleOptions,
    str,
    PuzzleBundle,
    int,
]:
    """Instantiate a puzzle and return the associated metadata."""

    cfg = _to_plain_dict(config)
    puzzle_name = cfg.get("name", "n-puzzle")
    puzzle_args = cfg.get("args", {}) or {}
    puzzle_size = str(cfg.get("size", "default"))
    seeds = cfg.get("seeds", [0])
    if isinstance(seeds, (list, tuple)):
        seeds_value = ",".join(str(seed) for seed in seeds)
    else:
        seeds_value = str(seeds)

    puzzle_opts = PuzzleOptions(
        puzzle=puzzle_name,
        puzzle_size=puzzle_size,
        puzzle_args=json.dumps(puzzle_args) if puzzle_args else "",
        hard=bool(cfg.get("hard", False)),
        seeds=seeds_value,
    )

    puzzle_bundle = puzzle_bundles[puzzle_opts.puzzle]
    input_args = dict(puzzle_args)
    if puzzle_opts.puzzle_size != "default":
        input_args["size"] = int(puzzle_opts.puzzle_size)

    if puzzle_opts.hard and puzzle_bundle.puzzle_hard is not None:
        puzzle_callable = puzzle_bundle.puzzle_hard
    else:
        puzzle_callable = puzzle_bundle.puzzle

    if isinstance(puzzle_callable, WorldModelPuzzleConfig):
        puzzle_instance = puzzle_callable.callable(path=puzzle_callable.path, **input_args)
    elif isinstance(puzzle_callable, PuzzleConfig):
        puzzle_instance = puzzle_callable.callable(
            initial_shuffle=puzzle_callable.initial_shuffle,
            **input_args,
        )
    elif puzzle_callable is None:
        raise ValueError(
            f"Puzzle type for '{puzzle_name}'{' (hard)' if puzzle_opts.hard else ''} is not defined."
        )
    else:
        puzzle_instance = puzzle_callable(**input_args)

    shuffle_override = cfg.get("shuffle_length")
    shuffle_length = (
        int(shuffle_override)
        if shuffle_override is not None
        else puzzle_bundle.shuffle_length
    )

    return puzzle_instance, puzzle_opts, puzzle_name, puzzle_bundle, shuffle_length


def build_dist_train_options(config: Mapping[str, Any] | DictConfig | None) -> DistTrainOptions:
    cfg = _to_plain_dict(config)
    return DistTrainOptions(**cfg)


def build_dist_eval_options(config: Mapping[str, Any] | DictConfig | None) -> EvalOptions:
    cfg = _to_plain_dict(config)
    return EvalOptions(**cfg)


def build_dist_q_options(config: Mapping[str, Any] | DictConfig | None) -> DistQFunctionOptions:
    cfg = _to_plain_dict(config)
    return DistQFunctionOptions(**cfg)


def build_davi_heuristic(
    config: Mapping[str, Any] | DictConfig | None,
    puzzle_bundle: PuzzleBundle,
    puzzle: Puzzle,
    puzzle_name: str,
    reset: bool,
) -> Tuple[NeuralHeuristicBase, NeuralCallableConfig]:
    cfg = _to_plain_dict(config)

    base_config = puzzle_bundle.heuristic_nn_config
    if base_config is None:
        raise ValueError(f"Neural heuristic not available for puzzle '{puzzle_name}'.")

    heuristic_config: NeuralCallableConfig = base_config.model_copy(deep=True)

    param_path = cfg.get("param_path")
    if param_path is None:
        param_path = heuristic_config.path_template.format(size=puzzle.size)

    neural_config_override = cfg.get("neural_config") or {}
    final_neural_config = heuristic_config.neural_config.copy() if heuristic_config.neural_config else {}
    final_neural_config.update(neural_config_override)
    heuristic_config.neural_config = final_neural_config

    heuristic: NeuralHeuristicBase = heuristic_config.callable(
        puzzle=puzzle,
        path=param_path,
        init_params=reset,
        **heuristic_config.neural_config,
    )
    return heuristic, heuristic_config


def build_qlearning_qfunction(
    config: Mapping[str, Any] | DictConfig | None,
    puzzle_bundle: PuzzleBundle,
    puzzle: Puzzle,
    puzzle_name: str,
    reset: bool,
) -> Tuple[NeuralQFunctionBase, NeuralCallableConfig]:
    cfg = _to_plain_dict(config)

    base_config = puzzle_bundle.q_function_nn_config
    if base_config is None:
        raise ValueError(f"Neural Q-function not available for puzzle '{puzzle_name}'.")

    q_config: NeuralCallableConfig = base_config.model_copy(deep=True)

    param_path = cfg.get("param_path")
    if param_path is None:
        param_path = q_config.path_template.format(size=puzzle.size)

    neural_config_override = cfg.get("neural_config") or {}
    final_neural_config = q_config.neural_config.copy() if q_config.neural_config else {}
    final_neural_config.update(neural_config_override)
    q_config.neural_config = final_neural_config

    qfunction: NeuralQFunctionBase = q_config.callable(
        puzzle=puzzle,
        path=param_path,
        init_params=reset,
        **q_config.neural_config,
    )
    return qfunction, q_config
