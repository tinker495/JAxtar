import click

from cli import astar, human_play, qstar, train


@click.group()
def main():
    pass


if __name__ == "__main__":
    main.add_command(human_play)
    main.add_command(astar)
    main.add_command(qstar)
    main.add_command(train)
    main()
