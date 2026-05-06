from pathlib import Path

import jax.numpy as jnp
import pytest
from puxle import SlidePuzzle

from helpers.visualization import build_path_steps_from_trace
from JAxtar.annotate import ACTION_DTYPE
from JAxtar.id_stars.search_base import IDSearchBase
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
    search_result = IDSearchBase.build(
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


@pytest.mark.parametrize(
    "runner_path",
    [
        "cli/search_runner.py",
        "cli/evaluation_runner.py",
    ],
)
def test_cli_runners_consume_solution_trace_interface(runner_path):
    source = (Path(__file__).parents[1] / runner_path).read_text()

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
