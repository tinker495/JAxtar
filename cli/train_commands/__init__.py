import click

from .train_command import davi, make_puzzle_dataset, qlearning


@click.group()
def train():
    pass


train.add_command(davi, name="davi")
train.add_command(qlearning, name="qlearning")
train.add_command(make_puzzle_dataset, name="make_puzzle_dataset")
