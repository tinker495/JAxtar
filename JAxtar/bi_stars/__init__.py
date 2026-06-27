"""JAxtar Bidirectional Stars Module."""

from __future__ import annotations

from _lazy_imports import lazy_dir, load_lazy_export

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


def __getattr__(name: str):
    return load_lazy_export(name, __name__, _EXPORTS, globals())


def __dir__() -> list[str]:
    return lazy_dir(globals(), __all__)
