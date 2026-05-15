"""Architecture guard for the Search Algorithm Catalog.

Locks the contract documented in CONTEXT.md "Search Algorithm Catalog": the
three CLI surfaces (`cli/commands.py`, `cli/eval_commands.py`,
`cli/benchmark_commands.py`) build their Click command sets by iterating the
Catalog. Drift between the Catalog and any CLI surface, or hand-written
`@click.command()` regressions in those surfaces, must fail this test.
"""

from __future__ import annotations

import re
from pathlib import Path

from config import SEARCH_ALGORITHM_CATALOG

from cli.benchmark_commands import benchmark
from cli.eval_commands import evaluation
from cli.main import cli

CATALOG_KEBAB = {entry.cli_subcommand for entry in SEARCH_ALGORITHM_CATALOG}


def test_catalog_entries_have_unique_python_ids():
    python_ids = [entry.python_id for entry in SEARCH_ALGORITHM_CATALOG]
    assert len(python_ids) == len(set(python_ids)), python_ids


def test_catalog_entries_have_unique_cli_subcommands():
    subs = [entry.cli_subcommand for entry in SEARCH_ALGORITHM_CATALOG]
    assert len(subs) == len(set(subs)), subs


def test_catalog_component_kinds_are_valid():
    for entry in SEARCH_ALGORITHM_CATALOG:
        assert entry.component_kind in ("heuristic", "qfunction"), entry


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


def test_cli_surface_files_have_no_hand_written_algorithm_dispatch():
    """Lock the shape: no `@click.command()` directly above an algorithm function
    in the three Catalog-iterating surface files. Catching this prevents the old
    copy-paste pattern from regrowing."""
    cli_dir = Path(__file__).resolve().parents[1] / "cli"
    pattern = re.compile(
        r"@click\.command\(\)\s*\n(?:@\w+(?:\([^)]*\))?\s*\n)+def\s+(?:astar|qstar|beam|qbeam|bi_|id_)",
    )
    for filename in ("commands.py", "eval_commands.py", "benchmark_commands.py"):
        source = (cli_dir / filename).read_text()
        match = pattern.search(source)
        assert match is None, (
            f"{filename} still hand-writes an algorithm-dispatch @click.command(); "
            "add the algorithm to SEARCH_ALGORITHM_CATALOG instead."
        )
