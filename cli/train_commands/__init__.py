import click

from .dist_train_command import davi, qlearning, spr_davi, spr_qlearning
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


distance_train.add_command(davi, name="davi")
distance_train.add_command(qlearning, name="qlearning")
distance_train.add_command(spr_davi, name="spr_davi")
distance_train.add_command(spr_qlearning, name="spr_qlearning")
world_model_train.add_command(make_puzzle_transition_dataset, name="make_transition_dataset")
world_model_train.add_command(make_puzzle_sample_data, name="make_sample_data")
world_model_train.add_command(make_puzzle_eval_trajectory, name="make_eval_trajectory")
world_model_train.add_command(train, name="train")
