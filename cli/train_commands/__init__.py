import click

from .ds_command import make_puzzle_sample_data, make_puzzle_transition_dataset
from .train_command import davi, qlearning


@click.group()
def train():
    pass


train.add_command(davi, name="davi")
train.add_command(qlearning, name="qlearning")
train.add_command(make_puzzle_transition_dataset, name="make_puzzle_transition_dataset")
train.add_command(make_puzzle_sample_data, name="make_puzzle_sample_data")
