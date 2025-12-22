import click

from .dist_train_command import heuristic_train_command, qfunction_train_command
from .world_model_ds_command import (
    make_puzzle_eval_trajectory,
    make_puzzle_sample_data,
    make_puzzle_transition_dataset,
)
from .world_model_train_command import train


@click.group()
def distance_train():
    pass


@click.group()
def world_model_train():
    pass


distance_train.add_command(heuristic_train_command, name="heuristic")
distance_train.add_command(qfunction_train_command, name="qfunction")
world_model_train.add_command(make_puzzle_transition_dataset, name="make_transition_dataset")
world_model_train.add_command(make_puzzle_sample_data, name="make_sample_data")
world_model_train.add_command(make_puzzle_eval_trajectory, name="make_eval_trajectory")
world_model_train.add_command(train, name="train")
