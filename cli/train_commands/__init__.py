import click

from .dist_train_command import dai, qlearning
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


distance_train.add_command(dai, name="dai")
distance_train.add_command(qlearning, name="qlearning")
world_model_train.add_command(make_puzzle_transition_dataset, name="make_transition_dataset")
world_model_train.add_command(make_puzzle_sample_data, name="make_sample_data")
world_model_train.add_command(make_puzzle_eval_trajectory, name="make_eval_trajectory")
world_model_train.add_command(train, name="train")
