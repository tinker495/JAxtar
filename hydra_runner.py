from __future__ import annotations

from typing import Any, Dict

import hydra
from omegaconf import DictConfig, OmegaConf

from cli.train_commands.dist_train_command import run_davi_training, run_qlearning_training
from hydra_support import (
    build_davi_heuristic,
    build_dist_eval_options,
    build_dist_q_options,
    build_dist_train_options,
    build_puzzle,
    build_qlearning_qfunction,
)


def _to_plain(config: Any) -> Any:
    if isinstance(config, DictConfig):
        return OmegaConf.to_container(config, resolve=True)
    return config


@hydra.main(version_base="1.3", config_path="conf", config_name="config")
def main(cfg: DictConfig) -> Dict[str, Any]:
    task_name = cfg.task.name

    puzzle, puzzle_opts, puzzle_name, puzzle_bundle, shuffle_length = build_puzzle(cfg.task.puzzle)
    train_options = build_dist_train_options(cfg.task.train)
    eval_options = build_dist_eval_options(cfg.task.eval)

    extra_config: Dict[str, Any] = {"hydra": OmegaConf.to_container(cfg, resolve=True)}
    task_extra = _to_plain(cfg.task.get("extra", {}))
    if isinstance(task_extra, dict) and task_extra:
        extra_config.update(task_extra)

    if task_name == "davi":
        heuristic, heuristic_config = build_davi_heuristic(
            cfg.task.heuristic,
            puzzle_bundle,
            puzzle,
            puzzle_name,
            reset=train_options.reset,
        )
        return run_davi_training(
            puzzle=puzzle,
            puzzle_opts=puzzle_opts,
            heuristic=heuristic,
            puzzle_name=puzzle_name,
            train_options=train_options,
            shuffle_length=shuffle_length,
            eval_options=eval_options,
            heuristic_config=heuristic_config,
            logger_run_name=cfg.task.run_name,
            extra_config=extra_config,
        )
    if task_name == "qlearning":
        qfunction, q_config = build_qlearning_qfunction(
            cfg.task.qfunction,
            puzzle_bundle,
            puzzle,
            puzzle_name,
            reset=train_options.reset,
        )
        q_options = build_dist_q_options(cfg.task.q_options)
        extra_config.setdefault("q_options", q_options.model_dump())
        return run_qlearning_training(
            puzzle=puzzle,
            puzzle_opts=puzzle_opts,
            qfunction=qfunction,
            puzzle_name=puzzle_name,
            train_options=train_options,
            shuffle_length=shuffle_length,
            with_policy=q_options.with_policy,
            eval_options=eval_options,
            q_config=q_config,
            logger_run_name=cfg.task.run_name,
            extra_config=extra_config,
        )

    raise ValueError(f"Unsupported task '{task_name}'.")


if __name__ == "__main__":
    main()
