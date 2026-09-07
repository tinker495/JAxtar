import importlib.abc
import os
import sys
from pathlib import Path

import pytest

OPTIONAL_STACK = {
    "aim",
    "tensorboardX",
    "wandb",
}


class _BlockOptionalStack(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path, target=None):
        if fullname.split(".", 1)[0] in OPTIONAL_STACK:
            raise ImportError(f"blocked optional dependency: {fullname}")
        return None


def _purge_modules(monkeypatch, *module_prefixes: str) -> None:
    for module_name in list(sys.modules):
        if module_name.split(".", 1)[0] in OPTIONAL_STACK or any(
            module_name == prefix or module_name.startswith(f"{prefix}.")
            for prefix in module_prefixes
        ):
            monkeypatch.delitem(sys.modules, module_name, raising=False)


def test_base_cli_help_does_not_import_logging_backends(monkeypatch, capsys):
    _purge_modules(
        monkeypatch,
        "cli.main",
        "cli.benchmark_commands",
        "cli.evaluation_runner",
        "cli.train_commands",
        "cli.train_commands.dist_train_command",
        "helpers.logger",
    )
    monkeypatch.delenv("TF_CPP_MIN_LOG_LEVEL", raising=False)
    monkeypatch.syspath_prepend(".")
    sys.meta_path.insert(0, _BlockOptionalStack())
    try:
        from cli.main import cli

        with pytest.raises(SystemExit) as exc_info:
            cli(args=["--help"])
        assert exc_info.value.code == 0
        base_output = capsys.readouterr().out

        for args in (
            ["distance-train", "heuristic", "--help"],
            ["distance-train", "qfunction", "--help"],
        ):
            with pytest.raises(SystemExit) as exc_info:
                cli(args=args)
            assert exc_info.value.code == 0
    finally:
        sys.meta_path = [
            finder for finder in sys.meta_path if not isinstance(finder, _BlockOptionalStack)
        ]

    capsys.readouterr()
    assert "benchmark" in base_output
    assert "eval" not in base_output
    assert "TF_CPP_MIN_LOG_LEVEL" not in os.environ
    assert "cli.train_commands.dist_train_command" not in sys.modules


def test_noop_logger_does_not_import_logging_backends(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    _purge_modules(monkeypatch, "helpers.logger")
    sys.meta_path.insert(0, _BlockOptionalStack())
    try:
        from helpers.logger import NoOpLogger, create_logger

        logger = create_logger("none", "unused", {})
    finally:
        sys.meta_path = [
            finder for finder in sys.meta_path if not isinstance(finder, _BlockOptionalStack)
        ]

    assert isinstance(logger, NoOpLogger)
    checkpoint = Path(logger.log_dir) / "checkpoint.pkl"
    checkpoint.write_bytes(b"checkpoint")
    assert checkpoint.is_file()
