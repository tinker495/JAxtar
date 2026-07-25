"""Search builder configuration shared by CLI adapters and search core."""

from __future__ import annotations

from dataclasses import dataclass
from math import inf
from typing import Any


@dataclass(frozen=True, slots=True)
class SearchBuildSpec:
    """Shared build-time knobs accepted by every search builder."""

    pop_ratio: float = inf
    # Keys are stored in KEY_DTYPE=float16, so cost_weight is rounded to float16 before it
    # multiplies g. Any value in ~[0.99976, 1.0] collapses to exactly 1.0 (the 1-1e-6 default
    # included) and runs as unweighted, admissible search; only weights <= ~0.9997 take effect.
    # Realistic greedy weights (0.2, 0.6, 0.8) are unaffected.
    cost_weight: float = 1.0 - 1e-6
    # ID-only: rounds each IDA* threshold up onto a grid of this size. The exact ladder
    # (bound_step=0) advances by one float16 ULP per pass under a real-valued heuristic
    # and re-derives the whole tree each time. Ignored by astar/qstar/beam.
    bound_step: float = 1.0
    # ID-only: length of the per-node action history, and simultaneously a hard depth
    # cutoff (`flat_depth <= max_path_len` prunes children). It dominates the stack row
    # -- 256 costs 283 B/node on RubiksCube against 91 B at 64, i.e. 5.7 GB vs 1.8 GB of
    # stack store at EvalOptions' 2e7 node budget. Lower it only above the puzzle's
    # solution depth; too low silently turns solvable instances into `solved=False`.
    max_path_len: int = 256
    show_compile_time: bool = False
    warmup_inputs: tuple[Any, Any] | None = None
    emit_workload_signature: bool = False


DEFAULT_SEARCH_BUILD_SPEC = SearchBuildSpec()


def _require_no_workload_signature(spec: SearchBuildSpec) -> None:
    if spec.emit_workload_signature:
        raise ValueError(
            "emit_workload_signature is only supported by astar, astar_d, and qstar builders."
        )
