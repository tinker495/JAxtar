"""Benchmark Verification helpers for JAxtar evaluation runs.

This Module owns the Benchmark Verification facts consumed by EvaluationRunner:
normalised action labels, exact-match status, and verification errors. It does
not reconstruct paths, schedule sweeps, write artifacts, or render progress.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence


@dataclass(frozen=True)
class BenchmarkVerification:
    """Facts produced by Benchmark Verification for one evaluated path."""

    path_action_strings: list[str] | None = None
    matches_optimal_path: bool | None = None
    benchmark_verification_error: str | None = None


_NO_VERIFICATION = BenchmarkVerification()
_VERIFY_BENCHMARK: Any | None = None


def build_benchmark_action_strings(
    *,
    puzzle: Any,
    actual_actions: Sequence[int] | None,
) -> list[str] | None:
    """Return benchmark-facing action labels, falling back to action ids as strings."""
    if not actual_actions:
        return None

    action_to_string_fn = getattr(puzzle, "action_to_string", None)
    labels: list[str] = []
    for action_id in actual_actions:
        if action_to_string_fn is None:
            labels.append(str(action_id))
            continue
        try:
            labels.append(action_to_string_fn(action_id))
        except (ValueError, IndexError):
            labels.append(str(action_id))
    return labels


def _can_verify(benchmark: Any, benchmark_sample: Any, states: Sequence[Any] | None) -> bool:
    return (
        benchmark is not None
        and benchmark_sample is not None
        and bool(states)
        and hasattr(benchmark, "verify_solution")
    )


def verify_benchmark_path_with_strings(
    *,
    benchmark: Any,
    benchmark_sample: Any,
    states: Sequence[Any] | None,
    path_action_strings: Sequence[str] | None,
) -> BenchmarkVerification:
    """Run benchmark verification when all required facts are available."""
    if not _can_verify(benchmark, benchmark_sample, states):
        return _NO_VERIFICATION

    action_sequence = list(path_action_strings) if path_action_strings is not None else None
    try:
        verification_result = benchmark.verify_solution(
            benchmark_sample,
            states=states,
            action_sequence=action_sequence,
        )
    except (Exception) as exc:  # noqa: BLE001 - turn benchmark verifier failures into facts.
        return BenchmarkVerification(
            path_action_strings=action_sequence,
            matches_optimal_path=None,
            benchmark_verification_error=str(exc),
        )

    return BenchmarkVerification(
        path_action_strings=action_sequence,
        matches_optimal_path=verification_result,
        benchmark_verification_error=None,
    )


def verify_benchmark_path(
    *,
    benchmark: Any,
    puzzle: Any,
    benchmark_sample: Any,
    states: Sequence[Any] | None,
    actual_actions: Sequence[int] | None,
    path_action_strings: Sequence[str] | None = None,
) -> BenchmarkVerification:
    """Build action labels and verify a reconstructed benchmark path."""
    if not _can_verify(benchmark, benchmark_sample, states):
        return _NO_VERIFICATION

    action_strings = (
        list(path_action_strings)
        if path_action_strings is not None
        else build_benchmark_action_strings(puzzle=puzzle, actual_actions=actual_actions)
    )
    return verify_benchmark_path_with_strings(
        benchmark=benchmark,
        benchmark_sample=benchmark_sample,
        states=states,
        path_action_strings=action_strings,
    )


def init_verify_worker(benchmark: Any) -> None:
    """Initialise a process-pool worker with a benchmark object."""
    global _VERIFY_BENCHMARK
    _VERIFY_BENCHMARK = benchmark


def verify_solution_worker(
    args: tuple[Any, Sequence[Any], Sequence[str] | None]
) -> BenchmarkVerification:
    """Process-pool Adapter for Benchmark Verification."""
    benchmark_sample, states, action_sequence = args
    return verify_benchmark_path_with_strings(
        benchmark=_VERIFY_BENCHMARK,
        benchmark_sample=benchmark_sample,
        states=states,
        path_action_strings=action_sequence,
    )
