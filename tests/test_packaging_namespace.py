import os
import tomllib
from pathlib import Path
from types import SimpleNamespace

import pytest

from JAxtar import entrypoints


def _load_pyproject():
    return tomllib.loads(Path("pyproject.toml").read_text(encoding="utf-8"))


def test_distribution_excludes_generic_top_level_packages():
    config = _load_pyproject()

    scripts = config["project"]["scripts"]
    expected_scripts = {name: f"JAxtar.entrypoints:{name}" for name in scripts}
    expected_scripts["jaxtar"] = "JAxtar.entrypoints:cli"
    assert scripts == expected_scripts
    assert config["tool"]["setuptools"]["packages"]["find"]["include"] == ["JAxtar*"]


def test_external_dependencies_use_pinned_git_sources():
    sources = _load_pyproject()["tool"]["uv"]["sources"]

    for name in ("puxle", "xtructure"):
        assert sources[name]["git"].startswith("https://github.com/tinker495/")
        assert len(sources[name]["rev"]) == 40
        assert "path" not in sources[name]


def test_xtructure_base_dependency_does_not_force_an_accelerator():
    lock = tomllib.loads(Path("uv.lock").read_text(encoding="utf-8"))
    xtructure = next(package for package in lock["package"] if package["name"] == "xtructure")
    jax_requirement = next(
        dependency for dependency in xtructure["dependencies"] if dependency["name"] == "jax"
    )

    assert "extra" not in jax_requirement


def test_console_environment_is_set_before_command_import(monkeypatch):
    observed = {}

    def command():
        observed["called"] = True

    def fake_import_module(module_name):
        observed["module_name"] = module_name
        observed["matplotlib_backend"] = os.environ.get("MPLBACKEND")
        observed["cpp_log_level"] = os.environ.get("TF_CPP_MIN_LOG_LEVEL")
        return SimpleNamespace(command=command)

    monkeypatch.delenv("MPLBACKEND", raising=False)
    monkeypatch.delenv("TF_CPP_MIN_LOG_LEVEL", raising=False)
    monkeypatch.setattr(entrypoints, "import_module", fake_import_module)

    entrypoints._run("fake.module", "command")

    assert observed == {
        "called": True,
        "module_name": "fake.module",
        "matplotlib_backend": "Agg",
        "cpp_log_level": "3",
    }


def test_console_entrypoint_rejects_non_checkout_install(monkeypatch, tmp_path):
    installed_entrypoint = tmp_path / "site-packages" / "JAxtar" / "entrypoints.py"
    monkeypatch.setattr(entrypoints, "__file__", str(installed_entrypoint))

    with pytest.raises(SystemExit, match="checkout-only"):
        entrypoints._run("cli.main", "cli")
