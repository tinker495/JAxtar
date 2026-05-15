from pathlib import Path

import jax.numpy as jnp
import pytest
from puxle import SlidePuzzle

from helpers.visualization import build_path_steps_from_trace
from JAxtar.annotate import ACTION_DTYPE
from JAxtar.id_stars.search_base import IDSearchResult
from JAxtar.solution_trace import (
    SolutionTrace,
    action_pad_int,
    normalise_action_sequence,
)


def test_normalise_action_sequence_stops_at_padding():
    action_pad = action_pad_int(ACTION_DTYPE)

    assert normalise_action_sequence([2, 1, action_pad, 0], action_pad=action_pad) == (
        2,
        1,
    )


def test_id_search_base_solution_trace_is_host_side_and_requires_replay():
    puzzle = SlidePuzzle(size=2)
    action_pad = action_pad_int(ACTION_DTYPE)
    search_result = IDSearchResult.build(
        puzzle.State,
        capacity=8,
        action_size=puzzle.action_size,
        max_path_len=4,
    ).replace(
        solved=jnp.array(True),
        solution_actions_arr=jnp.array(
            [1, action_pad, 0, action_pad],
            dtype=ACTION_DTYPE,
        ),
    )

    trace = search_result.to_solution_trace()

    assert trace == SolutionTrace(
        solved=True,
        actions=(1,),
        states=None,
        costs=None,
        dists=None,
        requires_replay=True,
    )


def test_path_step_adapter_replays_when_trace_requires_replay():
    class ReplayPuzzle:
        def get_neighbours(self, solve_config, state, filled):
            return jnp.array([state + 1, state + 10]), jnp.array([1.0, 10.0])

    trace = SolutionTrace(solved=True, actions=(0, 1), requires_replay=True)

    steps = build_path_steps_from_trace(
        puzzle=ReplayPuzzle(),
        solve_config=None,
        initial_state=jnp.array(0),
        solution_trace=trace,
    )

    assert [int(step.state) for step in steps] == [0, 1, 11]
    assert [step.cost for step in steps] == [0.0, 1.0, 11.0]
    assert [step.action for step in steps] == [0, 1, None]


def test_path_step_adapter_uses_supplied_trace_without_replay():
    class NoReplayPuzzle:
        def get_neighbours(self, solve_config, state, filled):  # pragma: no cover
            raise AssertionError("supplied solution trace should avoid replay")

    trace = SolutionTrace(
        solved=True,
        actions=(1,),
        states=("start", "goal"),
        costs=(0.0, 2.0),
        dists=(3.0, 0.0),
        requires_replay=False,
    )

    steps = build_path_steps_from_trace(
        puzzle=NoReplayPuzzle(),
        solve_config=None,
        initial_state="ignored",
        solution_trace=trace,
    )

    assert [(step.state, step.cost, step.dist, step.action) for step in steps] == [
        ("start", 0.0, 3.0, 1),
        ("goal", 2.0, 0.0, None),
    ]


_REPO_ROOT = Path(__file__).parents[1]

_RUNNER_PATHS = sorted(
    str(p.relative_to(_REPO_ROOT)) for p in (_REPO_ROOT / "cli").glob("*_runner.py")
)
assert _RUNNER_PATHS, (
    "No cli/*_runner.py files found; the Solution Trace Interface seam guard "
    "would silently pass. If runners have moved, update this glob."
)


@pytest.mark.parametrize("runner_path", _RUNNER_PATHS)
def test_cli_runners_consume_solution_trace_interface(runner_path):
    source = (_REPO_ROOT / runner_path).read_text()

    forbidden = (
        "reconstruct_bidirectional_path",
        "search_result.solution_trace(",
        "solution_actions",
        "get_solved_path",
        "build_path_steps_from_actions",
        "build_path_steps_from_nodes",
    )
    for phrase in forbidden:
        assert phrase not in source


def test_path_step_adapter_consumes_solution_trace_interface_only():
    """Path Step Adapter (helpers/visualization.py) must consume SolutionTrace
    through the public Solution Trace Interface only. It may import SolutionTrace
    for type annotations but must not call search-internal trace helpers or
    construct solved/unsolved SolutionTrace directly."""
    source = (_REPO_ROOT / "helpers/visualization.py").read_text()

    forbidden = (
        "_get_solved_path(",
        "_solution_actions(",
        "SolutionTrace(solved=True",
        "SolutionTrace(solved=False",
        "reconstruct_bidirectional_path",
    )
    for phrase in forbidden:
        assert phrase not in source, (
            f"helpers/visualization.py touched a search-internal trace path {phrase!r}; "
            "Path Step Adapter must consume SolutionTrace through the public "
            "Solution Trace Interface only (CONTEXT.md L156-158)."
        )


_VERIFICATION_GUARD_SUBTREES = (
    "cli",
    "helpers",
    "JAxtar",
    "heuristic",
    "qfunction",
    "world_model_puzzle",
    "train_util",
    "config",
    "neural_util",
    "tests",
)

_VERIFICATION_GUARD_ALLOWED = frozenset(
    {
        _REPO_ROOT / "cli/verification.py",
        _REPO_ROOT / "tests/test_cli_verification.py",
        # Self-reference: this guard contains the forbidden substring literal so
        # it can detect it elsewhere; exempting itself prevents a false positive.
        _REPO_ROOT / "tests/test_solution_trace.py",
    }
)


def test_benchmark_verification_not_constructed_outside_module():
    """Only the Verification Module (cli/verification.py) and its unit test may
    construct BenchmarkVerification directly. All other code must obtain
    verification facts through verify_benchmark_path() or
    benchmark_verification_from_exception() so the Module owns comparison rules
    (CONTEXT.md L119-122)."""
    offenders = []
    for subtree in _VERIFICATION_GUARD_SUBTREES:
        for py_file in (_REPO_ROOT / subtree).rglob("*.py"):
            if py_file in _VERIFICATION_GUARD_ALLOWED:
                continue
            if "BenchmarkVerification(" in py_file.read_text():
                offenders.append(str(py_file.relative_to(_REPO_ROOT)))
    assert not offenders, (
        "BenchmarkVerification(...) is constructed outside the Verification Module: "
        f"{offenders}. Use verify_benchmark_path() or "
        "benchmark_verification_from_exception() instead."
    )


def test_solution_trace_from_raw_unsolved_short_circuits():
    action_pad = action_pad_int(ACTION_DTYPE)
    assert (
        SolutionTrace.from_raw(
            solved=False,
            raw_actions=[1, 2, action_pad],
            action_pad=action_pad,
            states=("a",),
            costs=(0.0,),
            dists=(0.0,),
        )
        == SolutionTrace.unsolved()
    )


def test_solution_trace_from_raw_marks_replay_when_any_trace_data_missing():
    action_pad = action_pad_int(ACTION_DTYPE)
    actions_only = SolutionTrace.from_raw(
        solved=True,
        raw_actions=[1, action_pad],
        action_pad=action_pad,
    )
    assert actions_only.requires_replay is True
    assert actions_only.actions == (1,)
    assert actions_only.states is None

    states_no_costs = SolutionTrace.from_raw(
        solved=True,
        raw_actions=[1, action_pad],
        action_pad=action_pad,
        states=("a", "b"),
    )
    assert states_no_costs.requires_replay is True


def test_solution_trace_from_raw_skips_replay_when_all_trace_data_present():
    action_pad = action_pad_int(ACTION_DTYPE)
    complete = SolutionTrace.from_raw(
        solved=True,
        raw_actions=[1, action_pad],
        action_pad=action_pad,
        states=("a", "b"),
        costs=(0.0, 1.0),
        dists=(2.0, 0.0),
    )
    assert complete.requires_replay is False


_SEARCH_BASE_PATHS = [
    "JAxtar/stars/search_base.py",
    "JAxtar/beamsearch/search_base.py",
    "JAxtar/bi_stars/bi_search_base.py",
    "JAxtar/id_stars/search_base.py",
]


@pytest.mark.parametrize("search_base_path", _SEARCH_BASE_PATHS)
def test_search_base_exposes_to_solution_trace(search_base_path):
    """Every search algorithm Result must expose the public Solution Trace
    Interface (`def to_solution_trace(...)`) so the SearchAlgorithmResult
    Protocol holds statically. CLI / evaluation / visualization adapters call
    this and only this seam (CONTEXT.md L161)."""
    source = (_REPO_ROOT / search_base_path).read_text()
    assert "def to_solution_trace(" in source, (
        f"{search_base_path} does not expose `to_solution_trace(...)` — every "
        "search Result must implement the public Solution Trace Interface."
    )


@pytest.mark.parametrize("search_base_path", _SEARCH_BASE_PATHS)
def test_search_base_does_not_expose_public_trace_helpers(search_base_path):
    source = (Path(__file__).parents[1] / search_base_path).read_text()
    forbidden = (
        "    def get_solved_path(",
        "    def solution_actions(",
        "    def solution_trace(",
    )
    for phrase in forbidden:
        assert phrase not in source, (
            f"{search_base_path} still exposes a public trace helper {phrase.strip()!r}; "
            "the Solution Trace Interface is `to_solution_trace` only."
        )


@pytest.mark.parametrize("search_base_path", _SEARCH_BASE_PATHS)
def test_search_base_constructs_solution_trace_only_through_from_raw(search_base_path):
    """Solved SolutionTrace must come from `SolutionTrace.from_raw(...)`; the only
    direct-construction exception is `SolutionTrace.unsolved()`. This locks the
    Solution Replay Requirement policy as a single source of truth."""
    source = (Path(__file__).parents[1] / search_base_path).read_text()
    forbidden = (
        "SolutionTrace(solved=True",
        "SolutionTrace(solved=False",
    )
    for phrase in forbidden:
        assert phrase not in source, (
            f"{search_base_path} constructs SolutionTrace directly with {phrase!r}; "
            "use SolutionTrace.from_raw(...) for solved traces and SolutionTrace.unsolved() otherwise."
        )
