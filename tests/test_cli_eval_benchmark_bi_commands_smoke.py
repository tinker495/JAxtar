from click.testing import CliRunner

from cli.main import cli


def test_eval_bi_commands_help_smoke():
    runner = CliRunner()
    commands = [
        ["eval", "bi-astar", "--help"],
        ["eval", "bi-astar-d", "--help"],
        ["eval", "bi-qstar", "--help"],
    ]

    for argv in commands:
        result = runner.invoke(cli, argv)
        assert result.exit_code == 0, result.output
        assert result.exception is None
        assert "Traceback (most recent call last)" not in result.output


def test_benchmark_bi_commands_help_smoke():
    runner = CliRunner()
    commands = [
        ["benchmark", "bi-astar", "--help"],
        ["benchmark", "bi-astar-d", "--help"],
        ["benchmark", "bi-qstar", "--help"],
    ]

    for argv in commands:
        result = runner.invoke(cli, argv)
        assert result.exit_code == 0, result.output
        assert result.exception is None
        assert "Traceback (most recent call last)" not in result.output
