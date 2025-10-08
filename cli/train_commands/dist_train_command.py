import click
from puxle import Puzzle

from config.pydantic_models import (
    DistTrainOptions,
    EvalOptions,
    NeuralCallableConfig,
    PuzzleOptions,
    WBSDistTrainOptions,
)
from heuristic.neuralheuristic.neuralheuristic_base import NeuralHeuristicBase
from qfunction.neuralq.neuralq_base import NeuralQFunctionBase
from train_util.train_algo import (
    run_davi_training,
    run_qlearning_training,
    run_wbsdai_training,
    run_wbsdqi_training,
)

from ..options import (
    dist_heuristic_options,
    dist_puzzle_options,
    dist_qfunction_options,
    dist_train_options,
    eval_options,
    wbs_dist_train_options,
)


@click.command()
@dist_puzzle_options
@dist_train_options
@dist_heuristic_options
@eval_options
def davi(
    puzzle: Puzzle,
    puzzle_opts: PuzzleOptions,
    heuristic: NeuralHeuristicBase,
    puzzle_name: str,
    train_options: DistTrainOptions,
    shuffle_length: int,
    eval_options: EvalOptions,
    heuristic_config: NeuralCallableConfig,
    **kwargs,
):
    run_davi_training(
        puzzle=puzzle,
        puzzle_opts=puzzle_opts,
        heuristic=heuristic,
        puzzle_name=puzzle_name,
        train_options=train_options,
        shuffle_length=shuffle_length,
        eval_options=eval_options,
        heuristic_config=heuristic_config,
        **kwargs,
    )


@click.command()
@dist_puzzle_options
@dist_train_options
@dist_qfunction_options
@eval_options
def qlearning(
    puzzle: Puzzle,
    puzzle_opts: PuzzleOptions,
    qfunction: NeuralQFunctionBase,
    puzzle_name: str,
    train_options: DistTrainOptions,
    shuffle_length: int,
    with_policy: bool,
    eval_options: EvalOptions,
    q_config: NeuralCallableConfig,
    **kwargs,
):
    run_qlearning_training(
        puzzle=puzzle,
        puzzle_opts=puzzle_opts,
        qfunction=qfunction,
        puzzle_name=puzzle_name,
        train_options=train_options,
        shuffle_length=shuffle_length,
        with_policy=with_policy,
        eval_options=eval_options,
        q_config=q_config,
        **kwargs,
    )


@click.command()
@dist_puzzle_options
@wbs_dist_train_options
@dist_heuristic_options
@eval_options
def wbsdai(
    puzzle: Puzzle,
    puzzle_opts: PuzzleOptions,
    heuristic: NeuralHeuristicBase,
    puzzle_name: str,
    train_options: WBSDistTrainOptions,
    heuristic_config: NeuralCallableConfig,
    eval_options: EvalOptions,
    **kwargs,
):
    run_wbsdai_training(
        puzzle=puzzle,
        puzzle_opts=puzzle_opts,
        heuristic=heuristic,
        puzzle_name=puzzle_name,
        train_options=train_options,
        heuristic_config=heuristic_config,
        eval_options=eval_options,
        **kwargs,
    )


@click.command()
@dist_puzzle_options
@wbs_dist_train_options
@dist_qfunction_options
@eval_options
def wbsdqi(
    puzzle: Puzzle,
    puzzle_opts: PuzzleOptions,
    qfunction: NeuralQFunctionBase,
    puzzle_name: str,
    train_options: WBSDistTrainOptions,
    q_config: NeuralCallableConfig,
    eval_options: EvalOptions,
    **kwargs,
):
    run_wbsdqi_training(
        puzzle=puzzle,
        puzzle_opts=puzzle_opts,
        qfunction=qfunction,
        puzzle_name=puzzle_name,
        train_options=train_options,
        q_config=q_config,
        eval_options=eval_options,
        **kwargs,
    )
