from click.testing import CliRunner

from cli.main import cli


def test_search_commands_smoke_executes_without_error():
    runner = CliRunner()
    base_args = [
        "--puzzle",
        "n-puzzle",
        "--puzzle_args",
        '{"size":2}',
        "--seeds",
        "0",
        "--max_node_size",
        "64",
        "--batch_size",
        "16",
        "--vmap_size",
        "1",
    ]
    commands = [
        "astar",
        "astar-d",
        "bi-astar",
        "bi-astar-d",
        "beam",
        "qbeam",
        "qstar",
        "id-astar",
        "id-qstar",
        "bi-qstar",
    ]

    for command in commands:
        result = runner.invoke(cli, [command] + base_args)
        assert result.exit_code == 0, result.output
        assert result.exception is None
        assert "Traceback (most recent call last)" not in result.output
