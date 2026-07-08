"""Locks for the Expansion Trace Interface (CONTEXT.md)."""

import inspect
from pathlib import Path

import jax.numpy as jnp
import numpy as np

from JAxtar.beamsearch.search_base import BeamSearchResult
from JAxtar.bi_stars.bi_search_base import BiDirectionalSearchResult
from JAxtar.expansion_trace import ExpansionTrace
from JAxtar.id_stars.search_base import IDSearchResult
from JAxtar.stars.search_base import SearchResult

RESULT_CLASSES = (
    SearchResult,
    BiDirectionalSearchResult,
    IDSearchResult,
    BeamSearchResult,
)


def test_search_base_exposes_to_expansion_trace():
    for result_cls in RESULT_CLASSES:
        assert callable(getattr(result_cls, "to_expansion_trace", None)), result_cls
    for result_cls in (BiDirectionalSearchResult, IDSearchResult, BeamSearchResult):
        assert "return None" in inspect.getsource(result_cls.to_expansion_trace), result_cls


def test_from_raw_masks_to_expanded_nodes():
    trace = ExpansionTrace.from_raw(
        pop_generation=jnp.array([-1, 0, 2, -1, 1]),
        cost=jnp.array([9.0, 1.0, 3.0, 9.0, 2.0]),
        dist=jnp.array([9.0, 4.0, 6.0, 9.0, 5.0]),
        parent_indices=jnp.array([7, 0, 1, 7, 0]),
        solved_index=2,
    )
    assert isinstance(trace, ExpansionTrace)
    assert isinstance(trace.pop_generation, np.ndarray)
    np.testing.assert_array_equal(trace.pop_generation, [0, 2, 1])
    np.testing.assert_array_equal(trace.cost, [1.0, 3.0, 2.0])
    np.testing.assert_array_equal(trace.dist, [4.0, 6.0, 5.0])
    np.testing.assert_array_equal(trace.original_indices, [1, 2, 4])
    np.testing.assert_array_equal(trace.parent_indices, [0, 1, 0])
    assert trace.solved_index == 2


def test_from_raw_none_when_nothing_expanded_and_optional_defaults():
    assert (
        ExpansionTrace.from_raw(
            pop_generation=jnp.array([-1, -1]),
            cost=jnp.array([0.0, 0.0]),
            dist=jnp.array([0.0, 0.0]),
        )
        is None
    )
    trace = ExpansionTrace.from_raw(
        pop_generation=np.array([0]), cost=np.array([1.0]), dist=np.array([2.0])
    )
    assert trace.parent_indices is None and trace.solved_index is None


def test_cli_runners_do_not_touch_expansion_internals():
    cli_dir = Path(__file__).resolve().parents[1] / "cli"
    runner_files = sorted(cli_dir.glob("*_runner.py"))
    assert runner_files, "no CLI runner files found"
    for runner_file in runner_files:
        source = runner_file.read_text()
        for token in ("pop_generation", "hashtable", "hashidx"):
            assert token not in source, f"{runner_file.name} leaks {token}"
