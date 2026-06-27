from pathlib import Path

import pytest

from cli.verification import (
    BenchmarkVerification,
    benchmark_verification_from_exception,
    build_benchmark_action_strings,
    verify_benchmark_path,
)


class LabelPuzzle:
    def action_to_string(self, action_id):
        if action_id == 9:
            raise ValueError("bad action")
        if action_id == 8:
            raise IndexError("bad index")
        return f"act-{action_id}"


class RecordingBenchmark:
    def __init__(self, result):
        self.result = result
        self.calls = []

    def verify_solution(self, sample, *, states=None, action_sequence=None):
        self.calls.append((sample, states, action_sequence))
        return self.result


class RaisingBenchmark:
    def verify_solution(self, sample, *, states=None, action_sequence=None):
        raise RuntimeError("verification exploded")


def test_build_benchmark_action_strings_uses_puzzle_labels_with_fallbacks():
    assert build_benchmark_action_strings(
        puzzle=LabelPuzzle(),
        actual_actions=[1, 9, 8],
    ) == ["act-1", "9", "8"]


def test_build_benchmark_action_strings_returns_none_without_actions():
    assert build_benchmark_action_strings(puzzle=LabelPuzzle(), actual_actions=[]) is None


@pytest.mark.parametrize("result", [True, False, None])
def test_verify_benchmark_path_preserves_three_valued_result(result):
    benchmark = RecordingBenchmark(result)
    states = [object(), object()]

    verification = verify_benchmark_path(
        benchmark=benchmark,
        puzzle=LabelPuzzle(),
        benchmark_sample="sample-1",
        states=states,
        actual_actions=[1, 2],
    )

    assert verification == BenchmarkVerification(
        path_action_strings=["act-1", "act-2"],
        matches_optimal_path=result,
        benchmark_verification_error=None,
    )
    assert benchmark.calls == [("sample-1", states, ["act-1", "act-2"])]


def test_verify_benchmark_path_converts_verifier_exception_to_fact():
    verification = verify_benchmark_path(
        benchmark=RaisingBenchmark(),
        puzzle=LabelPuzzle(),
        benchmark_sample="sample-1",
        states=[object()],
        actual_actions=[1],
    )

    assert verification.path_action_strings == ["act-1"]
    assert verification.matches_optimal_path is None
    assert verification.benchmark_verification_error == "verification exploded"


@pytest.mark.parametrize(
    ("benchmark", "benchmark_sample", "states"),
    [
        (None, "sample-1", [object()]),
        (RecordingBenchmark(True), None, [object()]),
        (RecordingBenchmark(True), "sample-1", []),
    ],
)
def test_verify_benchmark_path_noops_when_required_facts_are_missing(
    benchmark,
    benchmark_sample,
    states,
):
    verification = verify_benchmark_path(
        benchmark=benchmark,
        puzzle=LabelPuzzle(),
        benchmark_sample=benchmark_sample,
        states=states,
        actual_actions=[1],
    )

    assert verification == BenchmarkVerification()
    if isinstance(benchmark, RecordingBenchmark):
        assert benchmark.calls == []


def test_benchmark_verification_from_exception_wraps_message():
    verification = benchmark_verification_from_exception(RuntimeError("boom"))
    assert verification == BenchmarkVerification(benchmark_verification_error="boom")


def test_evaluation_runner_does_not_reach_past_verification_module_seam():
    """EvaluationRunner consumes the Verification Module's small Interface only.

    The Module exposes BenchmarkVerification (type), build_benchmark_action_strings,
    verify_benchmark_path, and benchmark_verification_from_exception. Reaching for
    any other symbol means a
    seam leak that future Verification Module refactors must work around.
    """
    source = (Path(__file__).parents[1] / "cli/evaluation_runner.py").read_text()
    forbidden = (
        "verify_benchmark_path_with_strings",
        "_verify_benchmark_path_with_strings",
        "from .verification import _",
    )
    for phrase in forbidden:
        assert phrase not in source, (
            f"cli/evaluation_runner.py reaches past the Verification Module seam ({phrase!r}); "
            "use `verify_benchmark_path(...)` or `benchmark_verification_from_exception(...)` instead."
        )
