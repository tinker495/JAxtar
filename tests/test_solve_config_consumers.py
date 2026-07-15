from io import StringIO

import jax
import jax.numpy as jnp
from puxle import DotKnot, PDDL, SlidePuzzle, TSP
from rich.console import Console
from rich.text import Text

from helpers.visualization import build_seed_setup_panel
from heuristic.pddl_heuristic import PDDLHeuristic
from heuristic.slidepuzzle_heuristic import SlidePuzzleHeuristic
from heuristic.tsp_heuristic import TSPHeuristic
from neural_util.model_preprocessing import SlidePuzzlePreProcessMixin
from world_model_puzzle.world_model_ds import create_sample_data


def _render_setup(puzzle) -> str:
    solve_config, state = puzzle.get_inits(jax.random.PRNGKey(0))
    panel = build_seed_setup_panel(
        puzzle=puzzle,
        has_goal_data=puzzle.has_goal_data,
        solve_config=solve_config,
        state=state,
        dist_text=Text("0"),
        seed=0,
    )
    output = StringIO()
    Console(file=output, width=200, color_system=None).print(panel)
    return output.getvalue()


def test_target_heuristic_and_neural_preprocessing_read_goal_spec():
    puzzle = SlidePuzzle(size=2)
    solve_config = puzzle.get_solve_config(jax.random.PRNGKey(0))

    assert float(SlidePuzzleHeuristic(puzzle).distance(solve_config, solve_config.GoalSpec)) == 0

    processor = SlidePuzzlePreProcessMixin.__new__(SlidePuzzlePreProcessMixin)
    processor.puzzle = puzzle
    processor.size_square = puzzle.size**2
    processor.is_fixed = False
    current = puzzle.State.from_unpacked(board=jnp.array([1, 2, 0, 3], dtype=jnp.uint8))
    alternate_goal = puzzle.State.from_unpacked(board=jnp.array([1, 0, 3, 2], dtype=jnp.uint8))
    alternate_config = puzzle.SolveConfig(
        InstanceContext=solve_config.InstanceContext,
        GoalSpec=alternate_goal,
    )

    assert not jnp.array_equal(
        processor.pre_process(solve_config, current),
        processor.pre_process(alternate_config, current),
    )


def test_tsp_heuristic_reads_instance_context():
    puzzle = TSP(size=3)
    distance_matrix = jnp.array(
        [[0, 2, 3], [2, 0, 1], [3, 1, 0]],
        dtype=jnp.float32,
    )
    solve_config = puzzle.SolveConfig(
        InstanceContext=puzzle.InstanceContext(
            points=jnp.zeros((3, 2), dtype=jnp.float32),
            distance_matrix=distance_matrix,
            start=jnp.uint16(0),
        ),
        GoalSpec=puzzle.GoalSpec(),
    )
    current = puzzle.State.from_unpacked(
        mask=jnp.array([True, False, False]),
        point=jnp.uint16(0),
    )

    assert float(TSPHeuristic(puzzle).distance(solve_config, current)) == 5


def test_pddl_goal_mask_and_generic_display_use_goal_data():
    pddl = PDDL.from_preset("blocksworld", problem_basename="bw-S-01")
    solve_config, state = pddl.get_inits(jax.random.PRNGKey(0))
    expected = jnp.logical_and(
        solve_config.GoalSpec.GoalMask,
        jnp.logical_not(state.unpacked_atoms),
    ).sum()

    assert float(PDDLHeuristic(pddl).distance(solve_config, state)) == float(expected)
    assert not pddl.has_target and pddl.has_goal_data
    assert "Goal" in _render_setup(pddl)

    for puzzle in (TSP(size=3), DotKnot(size=4, color_num=2)):
        assert not puzzle.has_goal_data
        assert "Goal" not in _render_setup(puzzle)


def test_world_model_sample_data_extracts_goal_spec():
    puzzle = SlidePuzzle(size=2)
    key = jax.random.PRNGKey(0)
    targets, _ = create_sample_data(puzzle, shuffle_parallel=2, key=key)
    solve_configs, _ = jax.vmap(puzzle.get_inits)(jax.random.split(key, 2))

    assert jnp.array_equal(targets.board, solve_configs.GoalSpec.board)
