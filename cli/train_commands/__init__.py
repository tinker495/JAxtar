import click

from .train_command import davi, qlearning


@click.group()
def train():
    pass


train.add_command(davi, name="davi")
train.add_command(qlearning, name="qlearning")
