from types import SimpleNamespace

import pytest

import cli.benchmark_commands as benchmark_commands
import cli.options as cli_options
from cli.main import cli


def test_benchmark_bi_commands_expose_exact_and_generated_modes(capsys):
    commands = [
        ["benchmark", "bi-astar", "--help"],
        ["benchmark", "bi-astar-d", "--help"],
        ["benchmark", "bi-qstar", "--help"],
    ]

    for argv in commands:
        with pytest.raises(SystemExit) as exc_info:
            cli(args=argv)
        assert exc_info.value.code == 0
        output = capsys.readouterr().out
        assert "--benchmark" in output
        assert "--puzzle" in output
        assert "Traceback (most recent call last)" not in output


def test_benchmark_target_routes_exact_and_generated_workloads(monkeypatch):
    monkeypatch.setattr(cli_options, "_benchmark_bundles", lambda: {"exact": object()})
    monkeypatch.setattr(cli_options, "_puzzle_bundles", lambda: {"generated": object()})

    with pytest.raises(ValueError, match="either --benchmark or --puzzle"):
        cli_options.resolve_benchmark(
            {
                "benchmark_key": "exact",
                "puzzle_key": "generated",
                "puzzle_args": "",
                "benchmark_args": "",
                "sample_ids": "",
                "sample_limit": None,
            }
        )

    with pytest.raises(ValueError, match="--puzzle-args requires --puzzle"):
        cli_options.resolve_benchmark(
            {
                "benchmark_key": None,
                "puzzle_key": None,
                "puzzle_args": '{"size": 5}',
                "benchmark_args": "",
                "sample_ids": "",
                "sample_limit": None,
            }
        )


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

    assert (
        benchmark_commands.benchmark(args=["astar", "--benchmark", exact_key, "--num-eval", "0"])
        is None
    )
    assert (
        benchmark_commands.benchmark(args=["astar", "--puzzle", puzzle_key, "--num-eval", "0"])
        is None
    )
    assert captured[0]["benchmark"] is exact
    assert captured[0]["puzzle"] == "exact-puzzle"
    assert captured[1]["benchmark"] is None
    assert captured[1]["puzzle"] == "generated-puzzle"
