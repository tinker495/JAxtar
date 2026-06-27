import jax.numpy as jnp
from puxle import SlidePuzzle

from helpers.path_steps import build_path_steps_from_trace
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
