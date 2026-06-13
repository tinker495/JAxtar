"""Search builder configuration shared by CLI adapters and search core."""

from __future__ import annotations

from dataclasses import dataclass
from math import inf
from typing import Any


@dataclass(frozen=True, slots=True)
class SearchBuildSpec:
    """Shared build-time knobs accepted by every search builder."""

    pop_ratio: float = inf
    cost_weight: float = 1.0 - 1e-6
    show_compile_time: bool = False
    warmup_inputs: tuple[Any, Any] | None = None
    emit_workload_signature: bool = False


DEFAULT_SEARCH_BUILD_SPEC = SearchBuildSpec()


def _require_no_workload_signature(spec: SearchBuildSpec) -> None:
    if spec.emit_workload_signature:
        raise ValueError(
            "emit_workload_signature is only supported by astar, astar_d, and qstar builders."
        )
