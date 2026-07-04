from __future__ import annotations

from importlib import import_module

import pytest


@pytest.mark.parametrize(
    ("module_name", "symbol_name"),
    [
        ("JAxtar.stars.astar", "astar_builder"),
        ("JAxtar.stars.astar_d", "astar_d_builder"),
        ("JAxtar.utils.batch_switcher", "variable_batch_switcher_builder"),
        ("cli.comparison_generator", "ComparisonGenerator"),
        ("JAxtar.beamsearch.heuristic_beam", "beam_builder"),
        ("JAxtar.id_stars.id_astar", "id_astar_builder"),
        ("JAxtar.id_stars.id_qstar", "id_qstar_builder"),
        ("JAxtar.beamsearch.q_beam", "qbeam_builder"),
        ("JAxtar.stars.qstar", "qstar_builder"),
    ],
)
def test_import_smoke(module_name: str, symbol_name: str):
    assert getattr(import_module(module_name), symbol_name) is not None
