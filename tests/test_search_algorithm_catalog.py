"""Search Algorithm Catalog contract tests."""

from __future__ import annotations

import inspect
import tomllib
from dataclasses import is_dataclass
from pathlib import Path

import pytest

from cli.benchmark_commands import benchmark
from cli.main import cli, search_test
from config import (
    SEARCH_ALGORITHM_CATALOG,
    SearchAlgorithmEntry,
    resolve_algorithm_for_component,
)
from JAxtar.search_build_spec import SearchBuildSpec

CATALOG_KEBAB = {entry.cli_subcommand for entry in SEARCH_ALGORITHM_CATALOG}


def test_catalog_entries_are_plain_dataclasses():
    assert is_dataclass(SearchAlgorithmEntry)
    assert not hasattr(SearchAlgorithmEntry, "model_validate")


def test_catalog_entries_have_unique_python_ids():
    python_ids = [entry.python_id for entry in SEARCH_ALGORITHM_CATALOG]
    assert len(python_ids) == len(set(python_ids)), python_ids


def test_catalog_entries_have_unique_cli_subcommands():
    subs = [entry.cli_subcommand for entry in SEARCH_ALGORITHM_CATALOG]
    assert len(subs) == len(set(subs)), subs


def test_catalog_component_kinds_are_valid():
    for entry in SEARCH_ALGORITHM_CATALOG:
        assert entry.component_kind in ("heuristic", "qfunction"), entry


def test_catalog_workload_signature_capability_matrix_is_explicit():
    supporting_ids = {
        entry.python_id for entry in SEARCH_ALGORITHM_CATALOG if entry.supports_workload_signature
    }
    assert supporting_ids == {"astar", "astar_d", "qstar"}


def test_catalog_builders_accept_search_build_spec():
    for entry in SEARCH_ALGORITHM_CATALOG:
        signature = inspect.signature(entry.builder_fn)
        parameters = list(signature.parameters.values())
        assert [parameter.name for parameter in parameters[:5]] == [
            "puzzle",
            parameters[1].name,
            "batch_size",
            "max_nodes",
            "spec",
        ]
        assert parameters[4].annotation is SearchBuildSpec


def test_catalog_resolves_adapter_payload_for_matching_component():
    run_label, builder_fn, extra_kwargs = resolve_algorithm_for_component("beam", "heuristic")
    assert run_label == "beam"
    assert builder_fn.__name__ == "beam_builder"
    assert extra_kwargs == {"node_metric_label": "Beam Slots"}


def test_catalog_rejects_cross_component_resolution():
    with pytest.raises(ValueError, match="expects 'qfunction'"):
        resolve_algorithm_for_component("qstar", "heuristic")


def test_search_test_surface_command_set_matches_catalog():
    """`test` must expose every Catalog algorithm as a kebab-case subcommand."""
    assert set(search_test.commands) == CATALOG_KEBAB


def test_search_commands_are_not_registered_at_top_level():
    assert "test" in cli.commands
    assert "eval" not in cli.commands
    assert not CATALOG_KEBAB.intersection(cli.commands)


def test_benchmark_surface_command_set_matches_catalog():
    """`benchmark` group must expose every Catalog algorithm as a kebab-case subcommand."""
    assert "compare" in benchmark.commands
    bench_names = set(benchmark.commands) - {"compare"}
    assert (
        bench_names == CATALOG_KEBAB
    ), f"benchmark subcommands {bench_names} != Catalog {CATALOG_KEBAB}"


def test_direct_script_set_matches_catalog():
    config = tomllib.loads(Path("pyproject.toml").read_text(encoding="utf-8"))
    expected = {"jaxtar", "benchmark", "distance_train"} | {
        entry.python_id for entry in SEARCH_ALGORITHM_CATALOG
    }

    assert set(config["project"]["scripts"]) == expected
