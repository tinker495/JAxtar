"""Search Click commands generated from the Search Algorithm Catalog.

See CONTEXT.md "Search Algorithm Catalog". The 10 algorithm-specific commands
(`astar`, `astar_d`, `bi_astar`, ...) are built by iterating
`SEARCH_ALGORITHM_CATALOG` and applying a single surface-specific factory.
"""

from __future__ import annotations

import click

from config.algorithm_registry import SEARCH_ALGORITHM_CATALOG, SearchAlgorithmEntry
from helpers import heuristic_dist_format, qfunction_dist_format

from .options import (
    heuristic_options,
    puzzle_options,
    qfunction_options,
    search_options,
    visualize_options,
)
from .search_runner import run_search_command


def _build_search_command(entry: SearchAlgorithmEntry) -> click.Command:
    component_dec = heuristic_options if entry.component_kind == "heuristic" else qfunction_options
    search_dec = search_options(variant="beam") if entry.is_beam else search_options

    def inner(**kwargs):
        component = kwargs[entry.component_kind]
        if entry.component_kind == "heuristic":
            dist_fn = component.distance
            dist_format = heuristic_dist_format
        else:
            dist_fn = component.q_value
            dist_format = qfunction_dist_format
        run_search_command(
            kwargs["puzzle"],
            kwargs["puzzle_name"],
            kwargs.get("seeds"),
            kwargs["search_options"],
            kwargs["visualize_options"],
            entry.builder_fn,
            entry.component_kind,
            component,
            dist_fn,
            dist_format,
            entry.search_title,
        )

    inner = visualize_options(inner)
    inner = component_dec(inner)
    inner = search_dec(inner)
    inner = puzzle_options(inner)
    return click.command(name=entry.cli_subcommand)(inner)


SEARCH_COMMANDS: tuple[click.Command, ...] = tuple(
    _build_search_command(entry) for entry in SEARCH_ALGORITHM_CATALOG
)


__all__ = ["SEARCH_COMMANDS"]
