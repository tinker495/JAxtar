"""Tyro search commands generated from the Search Algorithm Catalog."""

from dataclasses import asdict
from functools import partial

import tyro

from config.algorithm_registry import SEARCH_ALGORITHM_CATALOG, SearchAlgorithmEntry
from helpers import heuristic_dist_format, qfunction_dist_format

from .options import (
    HeuristicSearchArgs,
    QFunctionSearchArgs,
    resolve_heuristic,
    resolve_puzzle,
    resolve_qfunction,
    resolve_search,
    resolve_visualize,
)
from .runtime import run_cli
from .search_runner import run_search_command

search_app = tyro.extras.SubcommandApp()


def _build_search_command(entry: SearchAlgorithmEntry):
    schema = HeuristicSearchArgs if entry.component_kind == "heuristic" else QFunctionSearchArgs

    def run(options):
        kwargs = resolve_puzzle(asdict(options))
        kwargs = resolve_search(kwargs, variant="beam" if entry.is_beam else "default")
        resolver = resolve_heuristic if entry.component_kind == "heuristic" else resolve_qfunction
        kwargs = resolve_visualize(resolver(kwargs))
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

    run.__annotations__ = {"options": schema}
    run.__name__ = entry.python_id
    run.__doc__ = entry.search_title
    search_app.command(run, name=entry.cli_subcommand)
    return partial(run_cli, partial(tyro.cli, run), prog=entry.cli_subcommand)


_SEARCH_COMMANDS_BY_ID = {
    entry.python_id: _build_search_command(entry) for entry in SEARCH_ALGORITHM_CATALOG
}
globals().update(_SEARCH_COMMANDS_BY_ID)
SEARCH_COMMANDS = tuple(_SEARCH_COMMANDS_BY_ID.values())

__all__ = ["SEARCH_COMMANDS", "search_app", *_SEARCH_COMMANDS_BY_ID]
