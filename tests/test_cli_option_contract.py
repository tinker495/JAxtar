from types import SimpleNamespace

import pytest

import cli.options as cli_options
from cli.main import cli
from config import puzzle_bundles
from config.pydantic_models import SearchOptions


def test_search_presets_preserve_unset_fields_and_parse_numeric_aliases(monkeypatch):
    captured = []
    bundle = puzzle_bundles["n-puzzle"]
    monkeypatch.setattr(
        bundle,
        "search_options_configs",
        {"custom": SearchOptions(batch_size=32, max_node_size=128, cost_weight=0.25)},
    )
    monkeypatch.setattr("cli.commands.run_search_command", lambda *args: captured.append(args))
    common = ["test", "astar", "--puzzle_args", '{"size": 2}', "--search-preset", "custom"]
    for extra in ([], ["-b", "2^6", "-m", "1K", "-w", "0"]):
        assert cli(args=common + extra) is None
    assert captured[0][3].batch_size == 32
    assert captured[0][3].max_node_size == 128
    assert captured[0][3].cost_weight == 0.25
    assert captured[1][3].batch_size == 64
    assert captured[1][3].max_node_size == 1000
    assert captured[1][3].cost_weight == 0


def test_search_optional_flag_preserves_preset_until_explicit(monkeypatch):
    captured = []
    bundle = puzzle_bundles["n-puzzle"]
    monkeypatch.setattr(
        bundle,
        "search_options_configs",
        {"custom": SearchOptions(show_compile_time=False, profile=True)},
    )
    monkeypatch.setattr("cli.commands.run_search_command", lambda *args: captured.append(args))
    common = ["test", "astar", "--puzzle-args", '{"size": 2}', "--search-preset", "custom"]

    assert cli(args=common) is None
    assert cli(args=common + ["--show-compile-time"]) is None

    assert captured[0][3].show_compile_time is False
    assert captured[0][3].profile is True
    assert captured[1][3].show_compile_time is True
    assert captured[1][3].profile is True


def test_legacy_hard_alias_is_preserved(monkeypatch):
    constructed = []
    bundle = puzzle_bundles["n-puzzle"]
    monkeypatch.setattr(
        bundle, "puzzle", lambda **_kwargs: constructed.append("normal") or object()
    )
    monkeypatch.setattr(
        bundle, "puzzle_hard", lambda **_kwargs: constructed.append("hard") or object()
    )
    monkeypatch.setattr(bundle, "heuristic", lambda _puzzle: SimpleNamespace(distance=lambda: 0))
    monkeypatch.setattr("cli.commands.run_search_command", lambda *_args: None)

    assert (
        cli(
            args=[
                "test",
                "astar",
                "-h",
                "--puzzle-args",
                '{"size": 3}',
                "--max-node-size",
                "64",
                "--batch-size",
                "16",
            ]
        )
        is None
    )
    assert constructed == ["hard"]


@pytest.mark.parametrize(
    ("kind", "preset", "component_key", "config_key", "runner_name"),
    [
        ("heuristic", "davi", "heuristic", "heuristic_config", "run_heuristic_training"),
        ("qfunction", "qlearning", "qfunction", "q_config", "run_qfunction_training"),
    ],
)
def test_distance_train_dispatches_resolved_options(
    monkeypatch, kind, preset, component_key, config_key, runner_name
):
    import cli.train_commands.dist_train_command as train_backend

    captured = []
    puzzle = object()
    bundle = type("Bundle", (), {"k_max": 7})()
    component = object()
    monkeypatch.setattr(
        cli_options,
        "_build_puzzle",
        lambda _options, *, default_hard: ("fake", bundle, puzzle),
    )
    monkeypatch.setattr(
        cli_options,
        "_setup_neural_component",
        lambda *_args, **_kwargs: {component_key: component, config_key: {"fake": True}},
    )
    monkeypatch.setattr(train_backend, runner_name, lambda **kwargs: captured.append(kwargs))

    assert (
        cli(
            args=[
                "distance-train",
                kind,
                "--preset",
                preset,
                "--steps",
                "1K",
                "--reset",
                "false",
                "--label",
                "diffusion",
            ]
        )
        is None
    )
    assert captured[0]["puzzle"] is puzzle
    assert captured[0][component_key] is component
    assert captured[0][config_key] == {"fake": True}
    assert captured[0]["k_max"] == 7
    assert captured[0]["train_options"].steps == 1000
    assert captured[0]["train_options"].reset is False
    assert captured[0]["train_options"].label == "diffusion"


@pytest.mark.parametrize(
    ("extra", "message"),
    [
        (["--batch-size", "bad"], "not a valid human-formatted integer"),
        (["--puzzle-args", "not-json"], "Expecting value"),
        (["--puzzle", "not-registered"], "invalid choice"),
    ],
)
def test_search_reports_invalid_cli_input(extra, message, capsys):
    with pytest.raises(SystemExit) as exc_info:
        cli(args=["test", "astar", *extra])

    assert exc_info.value.code == 2
    assert message in " ".join(capsys.readouterr().err.split())


@pytest.mark.parametrize(
    "options_type,field",
    [
        (cli_options.PuzzleArgs, "puzzle"),
        (cli_options.TrainPuzzleArgs, "puzzle"),
        (cli_options.BenchmarkArgs, "puzzle_key"),
    ],
)
def test_puzzle_choices_match_registry(options_type, field, capsys):
    import tyro

    with pytest.raises(SystemExit) as exc_info:
        tyro.cli(options_type, args=["--help"])
    assert exc_info.value.code == 0
    help_text = capsys.readouterr().out
    for name in puzzle_bundles:
        assert name in help_text
        parsed = tyro.cli(options_type, args=["--puzzle", name], console_outputs=False)
        assert getattr(parsed, field) == name

    with pytest.raises(SystemExit) as exc_info:
        tyro.cli(options_type, args=["--puzzle", "not-registered"], console_outputs=False)
    assert exc_info.value.code == 2
