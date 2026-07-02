"""Search Algorithm Catalog.

Single source of truth for the `(name → builder_fn, component_kind, beam variant,
surface labels)` mapping consumed by every CLI surface (search, eval, benchmark).
See CONTEXT.md "Search Algorithm Catalog".
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Literal

from JAxtar.beamsearch.heuristic_beam import beam_builder
from JAxtar.beamsearch.q_beam import qbeam_builder
from JAxtar.bi_stars.bi_astar import bi_astar_builder
from JAxtar.bi_stars.bi_astar_d import bi_astar_d_builder
from JAxtar.bi_stars.bi_qstar import bi_qstar_builder
from JAxtar.id_stars.id_astar import id_astar_builder
from JAxtar.id_stars.id_qstar import id_qstar_builder
from JAxtar.stars.astar import astar_builder
from JAxtar.stars.astar_d import astar_d_builder
from JAxtar.stars.qstar import qstar_builder

ComponentKind = Literal["heuristic", "qfunction"]


@dataclass(frozen=True, slots=True)
class SearchAlgorithmEntry:
    """One algorithm's facts shared across the three CLI surfaces."""

    python_id: str
    cli_subcommand: str
    builder_fn: Callable
    component_kind: ComponentKind
    search_title: str
    eval_description: str
    is_beam: bool = False
    node_metric_label: str | None = None
    supports_workload_signature: bool = False


SEARCH_ALGORITHM_CATALOG: tuple[SearchAlgorithmEntry, ...] = (
    SearchAlgorithmEntry(
        python_id="astar",
        cli_subcommand="astar",
        builder_fn=astar_builder,
        component_kind="heuristic",
        search_title="A* Search Configuration",
        eval_description="Evaluate a heuristic-driven A* search with optional parameter sweeps.",
        supports_workload_signature=True,
    ),
    SearchAlgorithmEntry(
        python_id="astar_d",
        cli_subcommand="astar-d",
        builder_fn=astar_d_builder,
        component_kind="heuristic",
        search_title="A* Deferred Search Configuration",
        eval_description=(
            "Evaluate a heuristic-driven A* Deferred search with optional parameter sweeps."
        ),
        supports_workload_signature=True,
    ),
    SearchAlgorithmEntry(
        python_id="bi_astar",
        cli_subcommand="bi-astar",
        builder_fn=bi_astar_builder,
        component_kind="heuristic",
        search_title="Bidirectional A* Search Configuration",
        eval_description=(
            "Evaluate a heuristic-driven bidirectional A* search " "with optional parameter sweeps."
        ),
    ),
    SearchAlgorithmEntry(
        python_id="bi_astar_d",
        cli_subcommand="bi-astar-d",
        builder_fn=bi_astar_d_builder,
        component_kind="heuristic",
        search_title="Bidirectional A* Deferred Search Configuration",
        eval_description=(
            "Evaluate a heuristic-driven bidirectional A* deferred search "
            "with optional parameter sweeps."
        ),
    ),
    SearchAlgorithmEntry(
        python_id="beam",
        cli_subcommand="beam",
        builder_fn=beam_builder,
        component_kind="heuristic",
        is_beam=True,
        search_title="Beam Search Configuration",
        eval_description="Evaluate a heuristic-driven beam search with optional parameter sweeps.",
        node_metric_label="Beam Slots",
    ),
    SearchAlgorithmEntry(
        python_id="qbeam",
        cli_subcommand="qbeam",
        builder_fn=qbeam_builder,
        component_kind="qfunction",
        is_beam=True,
        search_title="Q-beam Search Configuration",
        eval_description="Evaluate a Q-beam search with optional parameter sweeps.",
        node_metric_label="Beam Slots",
    ),
    SearchAlgorithmEntry(
        python_id="qstar",
        cli_subcommand="qstar",
        builder_fn=qstar_builder,
        component_kind="qfunction",
        search_title="Q* Search Configuration",
        eval_description="Evaluate a Q*-style search with optional parameter sweeps.",
        supports_workload_signature=True,
    ),
    SearchAlgorithmEntry(
        python_id="bi_qstar",
        cli_subcommand="bi-qstar",
        builder_fn=bi_qstar_builder,
        component_kind="qfunction",
        search_title="Bidirectional Q* Search Configuration",
        eval_description="Evaluate a bidirectional Q* search with optional parameter sweeps.",
    ),
    SearchAlgorithmEntry(
        python_id="id_astar",
        cli_subcommand="id-astar",
        builder_fn=id_astar_builder,
        component_kind="heuristic",
        search_title="IDA* Search Configuration",
        eval_description=(
            "Evaluate a heuristic-driven ID-A* search with optional parameter sweeps."
        ),
    ),
    SearchAlgorithmEntry(
        python_id="id_qstar",
        cli_subcommand="id-qstar",
        builder_fn=id_qstar_builder,
        component_kind="qfunction",
        search_title="ID-Q* Search Configuration",
        eval_description="Evaluate a Q*-style ID-Q* search with optional parameter sweeps.",
    ),
)


def get_algorithm_entry(python_id: str) -> SearchAlgorithmEntry:
    for entry in SEARCH_ALGORITHM_CATALOG:
        if entry.python_id == python_id:
            return entry
    raise KeyError(f"No Search Algorithm Catalog entry for python_id={python_id!r}")


def resolve_algorithm_for_component(
    python_id: str,
    component_kind: ComponentKind,
) -> tuple[str, Callable, dict]:
    entry = get_algorithm_entry(python_id)
    if entry.component_kind != component_kind:
        raise ValueError(
            f"Algorithm {python_id!r} expects {entry.component_kind!r}, not {component_kind!r}."
        )

    extra_kwargs: dict = {}
    if entry.node_metric_label is not None:
        extra_kwargs["node_metric_label"] = entry.node_metric_label
    return entry.python_id, entry.builder_fn, extra_kwargs


__all__ = [
    "ComponentKind",
    "SearchAlgorithmEntry",
    "SEARCH_ALGORITHM_CATALOG",
    "get_algorithm_entry",
    "resolve_algorithm_for_component",
]
