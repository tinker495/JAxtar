from types import SimpleNamespace

import click
from click.testing import CliRunner

import cli.benchmark_commands as benchmark_commands
import cli.options as cli_options
from cli.main import cli


def test_benchmark_bi_commands_expose_exact_and_generated_modes():
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
        assert "--benchmark" in result.output
        assert "--puzzle" in result.output
        assert "Traceback (most recent call last)" not in result.output


def test_benchmark_target_routes_exact_and_generated_workloads(monkeypatch):
    monkeypatch.setattr(cli_options, "_benchmark_bundles", lambda: {"exact": object()})
    monkeypatch.setattr(cli_options, "_puzzle_bundles", lambda: {"generated": object()})

    @click.command()
    @cli_options.benchmark_options
    def command(**kwargs):
        pass

    runner = CliRunner()
    result = runner.invoke(command, ["--benchmark", "exact", "--puzzle", "generated"])
    assert result.exit_code == 2
    assert "either --benchmark or --puzzle" in result.output

    result = runner.invoke(command, ["--puzzle-args", '{"size": 5}'])
    assert result.exit_code == 2
    assert "--puzzle-args requires --puzzle" in result.output


def test_benchmark_commands_route_both_targets_through_the_full_decorator_chain(monkeypatch):
    exact_key = next(iter(cli_options._benchmark_bundles()))
    puzzle_key = next(iter(cli_options._puzzle_bundles()))
    neural_config = SimpleNamespace(
        callable=lambda **kwargs: SimpleNamespace(metadata={}),
        param_path="unused.pkl",
    )
    exact = SimpleNamespace(puzzle="exact-puzzle")
    benchmark_bundle = SimpleNamespace(
        benchmark=lambda: exact,
        benchmark_args={},
        eval_options_configs={},
        heuristic_nn_configs={"default": neural_config},
    )
    puzzle_bundle = SimpleNamespace(
        puzzle=lambda: "generated-puzzle",
        puzzle_hard=None,
        eval_options_configs={},
        heuristic_nn_configs={"default": neural_config},
    )
    monkeypatch.setattr(cli_options, "_benchmark_bundles", lambda: {exact_key: benchmark_bundle})
    monkeypatch.setattr(cli_options, "_puzzle_bundles", lambda: {puzzle_key: puzzle_bundle})

    captured = []
    monkeypatch.setattr(
        benchmark_commands,
        "run_evaluation_sweep",
        lambda **kwargs: captured.append(kwargs),
    )

    runner = CliRunner()
    exact_result = runner.invoke(
        benchmark_commands.benchmark,
        ["astar", "--benchmark", exact_key, "--num-eval", "0"],
    )
    generated_result = runner.invoke(
        benchmark_commands.benchmark,
        ["astar", "--puzzle", puzzle_key, "--num-eval", "0"],
    )

    assert exact_result.exit_code == 0, exact_result.output
    assert generated_result.exit_code == 0, generated_result.output
    assert captured[0]["benchmark"] is exact
    assert captured[0]["puzzle"] == "exact-puzzle"
    assert captured[1]["benchmark"] is None
    assert captured[1]["puzzle"] == "generated-puzzle"
