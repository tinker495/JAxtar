import jax.numpy as jnp
import numpy as np

from cli.search_outcome import (
    attach_expansion_analysis,
    build_deferred_payload,
    build_evaluation_result_item,
    normalise_search_result,
    with_solution_path,
)
from JAxtar.expansion_trace import ExpansionTrace
from JAxtar.solution_trace import SolutionTrace


class UnidirectionalResult:
    solved = jnp.array(True)
    generated_size = jnp.array(7)
    solved_idx = "solved-slot"

    def get_cost(self, idx):
        assert idx == self.solved_idx
        return jnp.array(4.0)

    def to_solution_trace(self, *, puzzle=None):
        return SolutionTrace.from_raw(
            solved=True,
            raw_actions=[0],
            action_pad=999,
            states=("start", "goal"),
            costs=(0.0, 4.0),
            dists=(4.0, 0.0),
        )


class WorkloadResult(UnidirectionalResult):
    xtr_enabled = jnp.array(True)
    xtr_steps = jnp.array(2)
    xtr_cand_total = jnp.array(5)
    xtr_cand_valid = jnp.array(4)
    xtr_cand_unique = jnp.array(3)
    xtr_accept = jnp.array(2)


class Meeting:
    found = jnp.array(True)
    total_cost = jnp.array(9.0)


class BidirectionalResult:
    meeting = Meeting()
    forward = object()
    backward = object()
    total_generated = jnp.array(12)

    def to_solution_trace(self, *, puzzle=None):
        return SolutionTrace.from_raw(
            solved=True,
            raw_actions=[1],
            action_pad=999,
            states=("left", "right"),
            costs=(0.0, 9.0),
            dists=(None, None),
        )


class LabelPuzzle:
    def action_to_string(self, action_id):
        return f"act-{action_id}"


class BenchmarkSample:
    optimal_path_cost = 4.0
    optimal_action_sequence = [0]


class RecordingBenchmark:
    def __init__(self):
        self.calls = []

    def verify_solution(self, sample, *, states=None, action_sequence=None):
        self.calls.append((sample, states, action_sequence))
        return True


def test_search_outcome_normalises_unidirectional_result_facts():
    outcome = normalise_search_result(UnidirectionalResult())

    assert outcome.solved is True
    assert outcome.generated_size == 7
    assert outcome.solved_cost == 4.0


def test_search_outcome_normalises_bidirectional_result_facts():
    outcome = normalise_search_result(BidirectionalResult())

    assert outcome.solved is True
    assert outcome.generated_size == 12
    assert outcome.solved_cost == 9.0


def test_normalise_search_result_can_attach_workload_signature():
    outcome = normalise_search_result(WorkloadResult(), emit_workload_signature=True)

    assert outcome.workload_signature["xtr_steps"] == 2
    assert outcome.workload_signature["xtr_cand_total"] == 5


def test_solution_path_and_verification_facts_cross_outcome_seam():
    search_result = UnidirectionalResult()
    sample = BenchmarkSample()
    benchmark = RecordingBenchmark()
    outcome = with_solution_path(
        normalise_search_result(search_result),
        search_result,
        puzzle=LabelPuzzle(),
        solve_config=None,
        initial_state="ignored",
        benchmark=benchmark,
        benchmark_sample=sample,
    )

    result_item = build_evaluation_result_item(
        run_identifier=3,
        outcome=outcome,
        node_metric_label="Nodes Generated",
        benchmark_sample=sample,
    )
    payload = build_deferred_payload(
        result_item=result_item,
        outcome=outcome,
        solve_config=None,
        initial_state="ignored",
        benchmark_sample=sample,
    )

    assert outcome.path_actions == [0]
    assert result_item["path_cost"] == 4.0
    assert result_item["path_state_count"] == 2
    assert result_item["path_action_count"] == 1
    assert result_item["path_actions"] == [0]
    assert result_item["path_action_strings"] == ["act-0"]
    assert result_item["matches_optimal_path"] is True
    assert result_item["benchmark_optimal_path_cost"] == 4.0
    assert result_item["benchmark_optimal_action_sequence"] == [0]
    assert payload["states"] == ["start", "goal"]
    assert payload["actual_actions"] == [0]
    assert payload["verify_result"] is True
    assert benchmark.calls == [(sample, ["start", "goal"], ["act-0"])]


def test_attach_expansion_analysis_preserves_legacy_keys():
    trace = ExpansionTrace.from_raw(
        pop_generation=jnp.array([-1, 0, 1]),
        cost=jnp.array([9.0, 1.0, 2.0]),
        dist=jnp.array([9.0, 3.0, 4.0]),
        parent_indices=jnp.array([5, 0, 1]),
        solved_index=2,
    )

    class TracingResult:
        def to_expansion_trace(self):
            return trace

    result_item = {"expansion_analysis": None}
    attach_expansion_analysis(result_item, TracingResult())

    analysis = result_item["expansion_analysis"]
    assert set(analysis) == {
        "pop_generation",
        "cost",
        "dist",
        "original_indices",
        "parent_indices",
        "solved_index",
    }
    np.testing.assert_array_equal(analysis["pop_generation"], [0, 1])
    np.testing.assert_array_equal(analysis["original_indices"], [1, 2])
    np.testing.assert_array_equal(analysis["parent_indices"], [0, 1])
    assert analysis["solved_index"] == 2


def test_attach_expansion_analysis_leaves_item_untouched_without_trace():
    class NoTraceResult:
        def to_expansion_trace(self):
            return None

    result_item = {"expansion_analysis": None}
    attach_expansion_analysis(result_item, NoTraceResult())
    assert result_item["expansion_analysis"] is None
