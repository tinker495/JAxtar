"""Runner-facing search outcome normalisation.

This Module owns the outcome seam between CLI runners and algorithm-specific
search result implementations. Runners ask for solved / generated / cost /
trace / verification facts here instead of branching on Stars, Beam,
Bidirectional, or Iterative Deepening result internals.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any, Mapping, Sequence

import jax
import numpy as np
import xtructure.numpy as xnp

from helpers.path_steps import PathStep, build_path_steps_from_trace
from helpers.xtructure_signature import extract_xtructure_signature
from JAxtar.solution_trace import SolutionTrace

from .verification import BenchmarkVerification, verify_benchmark_path


@dataclass(frozen=True)
class SearchOutcome:
    """Normalised facts a runner needs after one search invocation."""

    solved: bool
    generated_size: int
    solved_cost: float | None
    workload_signature: Mapping[str, Any]
    solution_trace: SolutionTrace | None = None
    path_steps: tuple[PathStep, ...] = ()
    benchmark_verification: BenchmarkVerification | None = None

    @property
    def path_cost(self) -> float:
        if self.path_steps:
            return float(self.path_steps[-1].cost)
        if self.solved_cost is not None:
            return float(self.solved_cost)
        return 0.0

    @property
    def path_states(self) -> list[Any]:
        return [step.state for step in self.path_steps]

    @property
    def path_costs(self) -> list[float]:
        return [float(step.cost) for step in self.path_steps]

    @property
    def path_dists(self) -> list[float]:
        return [float(step.dist) if step.dist is not None else np.inf for step in self.path_steps]

    @property
    def path_actions(self) -> list[int]:
        return [int(step.action) for step in self.path_steps[:-1] if step.action is not None]

    @property
    def solution_state(self) -> Any | None:
        if not self.path_steps:
            return None
        return self.path_steps[-1].state


def normalise_search_result(
    search_result: Any,
    *,
    emit_workload_signature: bool = False,
) -> SearchOutcome:
    """Extract cheap runner facts while hiding Result implementation variants.

    This function intentionally does not materialise path steps. Callers can
    measure search time after this blocking normalisation, then opt into
    `with_solution_path(...)` for trace / artifact / verification facts.
    """

    solved = _result_solved(search_result)
    return SearchOutcome(
        solved=solved,
        generated_size=_generated_size(search_result),
        solved_cost=_solved_cost(search_result) if solved else None,
        workload_signature=(
            extract_xtructure_signature(search_result) if emit_workload_signature else {}
        ),
    )


def with_solution_path(
    outcome: SearchOutcome,
    search_result: Any,
    *,
    puzzle: Any,
    solve_config: Any,
    initial_state: Any,
    heuristic: Any | None = None,
    qfunction: Any | None = None,
    benchmark: Any | None = None,
    benchmark_sample: Any | None = None,
) -> SearchOutcome:
    """Attach Solution Trace, Path Step, and optional Benchmark Verification facts."""

    if not outcome.solved:
        return replace(outcome, solution_trace=SolutionTrace.unsolved(), path_steps=())

    solution_trace = search_result.to_solution_trace(puzzle=puzzle)
    path_steps = tuple(
        build_path_steps_from_trace(
            puzzle=puzzle,
            solve_config=solve_config,
            initial_state=initial_state,
            solution_trace=solution_trace,
            heuristic=heuristic,
            q_fn=qfunction,
        )
    )
    states = [step.state for step in path_steps]
    actions = [int(step.action) for step in path_steps[:-1] if step.action is not None]
    verification = None
    if benchmark is not None:
        verification = verify_benchmark_path(
            benchmark=benchmark,
            puzzle=puzzle,
            benchmark_sample=benchmark_sample,
            states=states,
            actual_actions=actions,
        )
    return replace(
        outcome,
        solution_trace=solution_trace,
        path_steps=path_steps,
        benchmark_verification=verification,
    )


def apply_benchmark_verification(
    result_item: dict,
    verification: BenchmarkVerification,
) -> None:
    """Apply Verification Module facts to an evaluation result artifact row."""

    if verification.path_action_strings is not None:
        result_item["path_action_strings"] = verification.path_action_strings
    if verification.benchmark_verification_error is not None:
        result_item["benchmark_verification_error"] = verification.benchmark_verification_error
    if (
        verification.matches_optimal_path is not None
        or verification.benchmark_verification_error is not None
    ):
        result_item["matches_optimal_path"] = verification.matches_optimal_path


def build_evaluation_result_item(
    *,
    run_identifier: int,
    outcome: SearchOutcome,
    node_metric_label: str,
    search_time_s: float = 0.0,
    benchmark_sample: Any | None = None,
) -> dict:
    """Build the per-sample artifact facts consumed by EvaluationRunner."""

    result_item = {
        "seed": run_identifier,
        "solved": outcome.solved,
        "search_time_s": search_time_s,
        "nodes_generated": outcome.generated_size,
        "node_metric_label": node_metric_label,
        "path_cost": outcome.path_cost if outcome.solved else 0,
        "path_analysis": None,
        "expansion_analysis": None,
        "path_state_count": len(outcome.path_steps) if outcome.path_steps else None,
        "path_action_count": max(0, len(outcome.path_steps) - 1) if outcome.path_steps else None,
        "matches_optimal_path": None,
        "path_actions": outcome.path_actions if outcome.path_steps else None,
        "path_action_strings": None,
        "benchmark_verification_error": None,
        "benchmark_has_optimal_action_sequence": False,
    }
    if outcome.workload_signature:
        result_item.update(outcome.workload_signature)
    if benchmark_sample is not None:
        result_item.update(_benchmark_reference_facts(benchmark_sample, run_identifier))
    if outcome.benchmark_verification is not None:
        apply_benchmark_verification(result_item, outcome.benchmark_verification)
    return result_item


def build_deferred_payload(
    *,
    result_item: dict,
    outcome: SearchOutcome,
    solve_config: Any,
    initial_state: Any,
    benchmark_sample: Any | None,
) -> dict:
    """Build deferred path / verification facts for EvaluationRunner finalisation."""

    payload = {
        "result_item": result_item,
        "solve_config": solve_config,
        "initial_state": initial_state,
        "benchmark_sample": benchmark_sample,
    }
    if not outcome.path_steps:
        return payload

    states = outcome.path_states
    costs = outcome.path_costs
    dists = outcome.path_dists
    verification = outcome.benchmark_verification
    payload.update(
        {
            "path_steps": list(outcome.path_steps),
            "states": states,
            "actual_actions": outcome.path_actions,
            "path_costs": costs,
            "path_dists": dists,
            "path_len": len(costs),
            "states_concat": _concatenate_states(states),
            "benchmark_verification": verification,
        }
    )
    if verification is not None:
        payload["path_action_strings"] = verification.path_action_strings
        payload["verify_result"] = verification.matches_optimal_path
        payload["verify_error"] = verification.benchmark_verification_error
    return payload


def _result_solved(search_result: Any) -> bool:
    if _is_bidirectional_result(search_result):
        return _ready_bool(search_result.meeting.found)
    return _ready_bool(search_result.solved)


def _generated_size(search_result: Any) -> int:
    if _is_bidirectional_result(search_result):
        return _as_int(search_result.total_generated)
    return _as_int(search_result.generated_size)


def _solved_cost(search_result: Any) -> float | None:
    if _is_bidirectional_result(search_result):
        return _as_float(search_result.meeting.total_cost)
    if not hasattr(search_result, "get_cost") or not hasattr(search_result, "solved_idx"):
        return None
    return _as_float(search_result.get_cost(search_result.solved_idx))


def _is_bidirectional_result(search_result: Any) -> bool:
    return (
        hasattr(search_result, "meeting")
        and hasattr(search_result, "forward")
        and hasattr(search_result, "backward")
    )


def _ready_bool(value: Any) -> bool:
    if hasattr(value, "block_until_ready"):
        value = value.block_until_ready()
    return bool(jax.device_get(value))


def _as_int(value: Any) -> int:
    return int(np.asarray(jax.device_get(value)))


def _as_float(value: Any) -> float:
    return float(np.asarray(jax.device_get(value)))


def _benchmark_reference_facts(benchmark_sample: Any, run_identifier: int) -> dict:
    facts: dict[str, Any] = {"benchmark_sample_id": run_identifier}

    optimal_path_cost = getattr(benchmark_sample, "optimal_path_cost", None)
    if optimal_path_cost is None:
        optimal_path_cost = getattr(benchmark_sample, "optimal_path_costs", None)
    if optimal_path_cost is not None:
        facts["benchmark_optimal_path_cost"] = float(optimal_path_cost)

    optimal_path = getattr(benchmark_sample, "optimal_path", None)
    if optimal_path is not None:
        optimal_state_count = len(optimal_path)
        facts["benchmark_optimal_path_state_count"] = optimal_state_count
        if optimal_state_count > 0:
            facts["benchmark_optimal_path_length"] = max(0, optimal_state_count - 1)

    optimal_action_sequence = getattr(benchmark_sample, "optimal_action_sequence", None)
    if optimal_action_sequence is not None:
        facts["benchmark_has_optimal_action_sequence"] = True
        if not isinstance(optimal_action_sequence, (list, tuple)):
            optimal_action_sequence = list(optimal_action_sequence)
        facts["benchmark_optimal_action_sequence"] = [
            action_val if isinstance(action_val, str) else int(action_val)
            for action_val in optimal_action_sequence
        ]
        facts["benchmark_optimal_action_count"] = len(facts["benchmark_optimal_action_sequence"])
    elif optimal_path is not None:
        facts["benchmark_optimal_action_count"] = max(0, len(optimal_path) - 1)

    return facts


def _concatenate_states(states: Sequence[Any]) -> Any | None:
    if not states:
        return None
    try:
        return xnp.concatenate(states)
    except (AttributeError, TypeError, ValueError):
        return None
