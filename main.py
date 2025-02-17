import click

from cli import astar, distance_train, human_play, qstar, world_model_train


@click.group()
def main():
    pass


if __name__ == "__main__":
    main.add_command(human_play)
    main.add_command(astar)
    main.add_command(qstar)
    main.add_command(distance_train)
    main.add_command(world_model_train)
    main()
