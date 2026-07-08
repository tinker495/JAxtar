"""Host-side solution trace exposed by search result modules."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Protocol

import numpy as np

from JAxtar.expansion_trace import ExpansionTrace


@dataclass(frozen=True)
class SolutionTrace:
    """Normalised host-side solution trace returned by search results."""

    solved: bool
    actions: tuple[int, ...]
    states: tuple[Any, ...] | None = None
    costs: tuple[float, ...] | None = None
    dists: tuple[float | None, ...] | None = None
    requires_replay: bool = False

    @classmethod
    def unsolved(cls) -> "SolutionTrace":
        return cls(solved=False, actions=())

    @classmethod
    def from_raw(
        cls,
        *,
        solved: bool,
        raw_actions: Iterable[Any],
        action_pad: int,
        states: tuple[Any, ...] | None = None,
        costs: tuple[float, ...] | None = None,
        dists: tuple[float | None, ...] | None = None,
    ) -> "SolutionTrace":
        """Build a SolutionTrace from algorithm-specific raw reconstruction data.

        Owns the Solution Replay Requirement policy: a trace requires replay
        when any of states / costs / dists is missing, so a Path Step Adapter
        can decide whether to replay actions through a puzzle to recover the
        missing facts.
        """
        if not solved:
            return cls.unsolved()
        actions = normalise_action_sequence(raw_actions, action_pad=action_pad)
        requires_replay = states is None or costs is None or dists is None
        return cls(
            solved=True,
            actions=actions,
            states=states,
            costs=costs,
            dists=dists,
            requires_replay=requires_replay,
        )


class SearchAlgorithmResult(Protocol):
    """Static seam every search Result class implements for CLI/evaluation adapters."""

    def to_solution_trace(self, *, puzzle: Any | None = None) -> SolutionTrace:
        ...

    def to_expansion_trace(self) -> ExpansionTrace | None:
        ...


def action_pad_int(action_dtype: Any) -> int:
    return int(np.iinfo(np.dtype(action_dtype)).max)


def normalise_action_sequence(
    actions: Iterable[Any],
    *,
    action_pad: int,
) -> tuple[int, ...]:
    normalised: list[int] = []
    for raw_action in actions:
        action = int(raw_action)
        if action == action_pad:
            break
        normalised.append(action)
    return tuple(normalised)
