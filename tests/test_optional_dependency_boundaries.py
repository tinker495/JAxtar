import importlib.abc
import sys

from click.testing import CliRunner


OPTIONAL_STACK = {
    "aim",
    "matplotlib",
    "pandas",
    "seaborn",
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


def test_base_cli_help_does_not_import_logging_or_plot_stack(monkeypatch):
    _purge_modules(
        monkeypatch,
        "cli.main",
        "cli.benchmark_commands",
        "cli.eval_commands",
        "cli.evaluation_runner",
        "cli.comparison_generator",
        "helpers.artifact_manager",
        "helpers.plots",
    )
    monkeypatch.syspath_prepend(".")
    sys.meta_path.insert(0, _BlockOptionalStack())
    try:
        from cli.main import cli

        result = CliRunner().invoke(cli, ["--help"])
    finally:
        sys.meta_path = [
            finder for finder in sys.meta_path if not isinstance(finder, _BlockOptionalStack)
        ]

    assert result.exit_code == 0, result.output
    assert "benchmark" in result.output
    assert "eval" in result.output


def test_noop_logger_does_not_import_logging_backends(monkeypatch):
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
