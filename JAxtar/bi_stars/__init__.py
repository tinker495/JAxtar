"""JAxtar Bidirectional Stars Module."""

from __future__ import annotations

import importlib
from typing import Any

__all__ = [
    "bi_astar_builder",
    "bi_astar_d_builder",
    "bi_qstar_builder",
    "reconstruct_bidirectional_path",
    "MeetingPoint",
    "BiDirectionalSearchResult",
    "BiLoopState",
    "BiLoopStateWithStates",
    "build_bi_search_result",
    "check_intersection",
    "update_meeting_point",
    "get_min_f_value",
    "bi_termination_condition",
]

_EXPORTS = {
    "bi_astar_builder": "JAxtar.bi_stars.bi_astar",
    "bi_astar_d_builder": "JAxtar.bi_stars.bi_astar_d",
    "bi_qstar_builder": "JAxtar.bi_stars.bi_qstar",
    "reconstruct_bidirectional_path": "JAxtar.bi_stars.bi_search_base",
    "MeetingPoint": "JAxtar.bi_stars.bi_search_base",
    "BiDirectionalSearchResult": "JAxtar.bi_stars.bi_search_base",
    "BiLoopState": "JAxtar.bi_stars.bi_search_base",
    "BiLoopStateWithStates": "JAxtar.bi_stars.bi_search_base",
    "build_bi_search_result": "JAxtar.bi_stars.bi_search_base",
    "check_intersection": "JAxtar.bi_stars.bi_search_base",
    "update_meeting_point": "JAxtar.bi_stars.bi_search_base",
    "get_min_f_value": "JAxtar.bi_stars.bi_search_base",
    "bi_termination_condition": "JAxtar.bi_stars.bi_search_base",
}


def __getattr__(name: str) -> Any:
    try:
        module_name = _EXPORTS[name]
    except KeyError as exc:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from exc
    value = getattr(importlib.import_module(module_name), name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))
