"""Unit + architecture guard for `JAxtar.parent_chain.walk_parent_chain`.

Locks the contract documented in CONTEXT.md "Parent Chain Walk":
- target-to-root return order
- cycle detection via max_steps budget
- single host-side parent walker site (no hand-written `_trace_*_to_*`
  duplicates).
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import pytest

from JAxtar.parent_chain import INVALID_PARENT_SENTINEL, walk_parent_chain


@dataclass(frozen=True)
class _StubHashIdx:
    index: int


@dataclass(frozen=True)
class _StubParent:
    hashidx: _StubHashIdx
    action: int


class _StubParentTable:
    """Minimal duck-type stand-in for `xtructure` parent record arrays."""

    def __init__(self, entries: dict[int, _StubParent]) -> None:
        self._entries = entries

    def __getitem__(self, idx: int) -> _StubParent:
        return self._entries[idx]


def _root(action: int = 0) -> _StubParent:
    return _StubParent(hashidx=_StubHashIdx(INVALID_PARENT_SENTINEL), action=action)


def test_walk_returns_single_root_immediately():
    table = _StubParentTable({7: _root()})
    indices, actions = walk_parent_chain(table, start_index=7, max_steps=4)
    assert indices == [7]
    assert actions == []


def test_walk_returns_target_to_root_order():
    table = _StubParentTable(
        {
            5: _StubParent(hashidx=_StubHashIdx(2), action=11),
            2: _StubParent(hashidx=_StubHashIdx(0), action=22),
            0: _root(),
        }
    )
    indices, actions = walk_parent_chain(table, start_index=5, max_steps=8)
    assert indices == [5, 2, 0]
    assert actions == [11, 22]


def test_walk_raises_on_cycle_overflow():
    table = _StubParentTable(
        {
            1: _StubParent(hashidx=_StubHashIdx(2), action=0),
            2: _StubParent(hashidx=_StubHashIdx(1), action=0),
        }
    )
    with pytest.raises(RuntimeError, match="cycle/corruption"):
        walk_parent_chain(table, start_index=1, max_steps=4)


def test_walk_respects_custom_sentinel():
    table = _StubParentTable(
        {
            3: _StubParent(hashidx=_StubHashIdx(-1), action=5),
        }
    )
    indices, actions = walk_parent_chain(table, start_index=3, max_steps=2, invalid_sentinel=-1)
    assert indices == [3]
    assert actions == []


def test_single_host_side_parent_walker_site():
    """Architecture guard: only `bi_search_base.py` may consume
    `walk_parent_chain`. Drift here means another module is host-walking
    parent pointers without going through the canonical helper."""
    repo_root = Path(__file__).resolve().parents[1]
    pattern = re.compile(r"\bwalk_parent_chain\b")
    consumers: list[str] = []
    for path in repo_root.rglob("*.py"):
        if path.name in {"parent_chain.py", "test_parent_chain.py"}:
            continue
        try:
            text = path.read_text()
        except OSError:
            continue
        if pattern.search(text):
            consumers.append(str(path.relative_to(repo_root)))

    assert consumers == ["JAxtar/bi_stars/bi_search_base.py"], consumers


def test_no_nested_trace_helper_regression():
    """Architecture guard: the old `_trace_root_to_target` /
    `_trace_target_to_root` nested helpers must not regrow inside
    `reconstruct_bidirectional_path`."""
    repo_root = Path(__file__).resolve().parents[1]
    source = (repo_root / "JAxtar" / "bi_stars" / "bi_search_base.py").read_text()
    forbidden = ("_trace_root_to_target", "_trace_target_to_root", "_u32_max")
    for name in forbidden:
        assert name not in source, (
            f"`{name}` reappeared in bi_search_base.py; consume "
            "`walk_parent_chain` from JAxtar.parent_chain instead."
        )
