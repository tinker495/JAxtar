"""Search Algorithm Catalog contract tests."""

from __future__ import annotations

import inspect
from dataclasses import is_dataclass

import pytest

from config import (
    SEARCH_ALGORITHM_CATALOG,
    SearchAlgorithmEntry,
    resolve_algorithm_for_component,
)

from cli.benchmark_commands import benchmark
from cli.eval_commands import evaluation
from cli.main import cli
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
    resolution = resolve_algorithm_for_component("beam", "heuristic")
    assert resolution.run_label == "beam"
    assert resolution.builder_fn.__name__ == "beam_builder"
    assert resolution.extra_kwargs == {"node_metric_label": "Beam Slots"}


def test_catalog_rejects_cross_component_resolution():
    with pytest.raises(ValueError, match="expects 'qfunction'"):
        resolve_algorithm_for_component("qstar", "heuristic")


def test_search_surface_command_set_matches_catalog():
    """Top-level CLI must expose every Catalog algorithm as a kebab-case command."""
    top_level_names = set(cli.commands.keys())
    missing = CATALOG_KEBAB - top_level_names
    assert not missing, f"Catalog algorithms missing from top-level CLI: {missing}"


def test_eval_surface_command_set_matches_catalog():
    """`eval` group must expose every Catalog algorithm as a kebab-case subcommand."""
    eval_names = set(evaluation.commands.keys())
    algorithm_names = eval_names - {"compare"}
    assert (
        algorithm_names == CATALOG_KEBAB
    ), f"eval algorithm subcommands {algorithm_names} != Catalog {CATALOG_KEBAB}"


def test_benchmark_surface_command_set_matches_catalog():
    """`benchmark` group must expose every Catalog algorithm as a kebab-case subcommand."""
    bench_names = set(benchmark.commands.keys())
    assert (
        bench_names == CATALOG_KEBAB
    ), f"benchmark subcommands {bench_names} != Catalog {CATALOG_KEBAB}"
