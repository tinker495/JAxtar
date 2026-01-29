"""JAxtar Bidirectional Stars Module.

This package contains bidirectional variants of A*/Q*.

Imports are kept lazy to reduce import-time coupling.
"""

from __future__ import annotations

from typing import Any


def __getattr__(name: str) -> Any:
    if name == "bi_astar_builder":
        from .bi_astar import bi_astar_builder

        return bi_astar_builder
    if name == "bi_astar_d_builder":
        from .bi_astar_d import bi_astar_d_builder

        return bi_astar_d_builder
    if name == "bi_qstar_builder":
        from .bi_qstar import bi_qstar_builder

        return bi_qstar_builder
    if name == "reconstruct_bidirectional_path":
        from .bi_search_base import reconstruct_bidirectional_path

        return reconstruct_bidirectional_path

    if name in {
        "MeetingPoint",
        "BiDirectionalSearchResult",
        "BiLoopState",
        "BiLoopStateWithStates",
        "build_bi_search_result",
        "check_intersection",
        "update_meeting_point",
        "get_min_f_value",
        "bi_termination_condition",
    }:
        from . import bi_search_base

        return getattr(bi_search_base, name)

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(
        {
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
        }
    )
