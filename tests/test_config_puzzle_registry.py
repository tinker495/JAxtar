from config import puzzle_registry
from puxle.world_model import trained_world_model_registry


def test_sized_bundle_injects_size_while_preserving_bundle_fields():
    base = puzzle_registry.puzzle_bundles["n-puzzle"]
    sized = puzzle_registry._sized_bundle(
        base,
        size=4,
        puzzle_cls=base.puzzle,
        hard_cls=None,
    )

    assert sized.puzzle.callable == base.puzzle
    assert sized.puzzle.kwargs == {"size": 4}
    assert sized.puzzle_hard is None
    assert sized.heuristic == base.heuristic
    assert sized.q_function == base.q_function
    assert sized.k_max == base.k_max


def test_generated_size_variants_are_registered_with_expected_settings():
    bundle = puzzle_registry.puzzle_bundles["rubikscube-3"]

    assert bundle.puzzle.callable.__name__ == "RubiksCube"
    assert bundle.puzzle.kwargs["size"] == 3
    assert bundle.eval_benchmark == "rubikscube-deepcubea"
    assert bundle.k_max == 26


def test_world_model_bundle_exists_for_optimized_sokoban():
    assert "sokoban_world_model_optimized" in puzzle_registry.puzzle_bundles
    bundle = puzzle_registry.puzzle_bundles["sokoban_world_model_optimized"]
    assert bundle.puzzle is trained_world_model_registry.SokobanWorldModelOptimized
    assert callable(bundle.puzzle)
    assert bundle.heuristic_nn_configs is not None
    assert bundle.q_function_nn_configs is not None
    assert any(k == "default" for k in bundle.heuristic_nn_configs)
    assert any(k == "default" for k in bundle.q_function_nn_configs)


def test_only_explicit_puzzle_args_reach_puzzle_constructor(monkeypatch):
    from click.testing import CliRunner

    from cli.main import cli

    constructed_with = []

    class FakePuzzle:
        size = 2

    class FakeHeuristic:
        def __init__(self, puzzle):
            self.puzzle = puzzle

        def distance(self, *_args):
            return 0.0

    bundle = puzzle_registry.puzzle_bundles["n-puzzle"]
    monkeypatch.setattr(
        bundle,
        "puzzle",
        lambda **kwargs: constructed_with.append(kwargs) or FakePuzzle(),
    )
    monkeypatch.setattr(bundle, "heuristic", FakeHeuristic)
    monkeypatch.setattr("cli.commands.run_search_command", lambda *_args: None)

    result = CliRunner().invoke(
        cli,
        [
            "astar-d",
            "-p",
            "n-puzzle",
            "--puzzle_args",
            '{"size": 2}',
            "--seeds",
            "7",
            "--batch_size",
            "16",
            "--max_node_size",
            "64",
        ],
    )

    assert result.exit_code == 0, result.output
    assert constructed_with == [{"size": 2}]


def test_neural_heuristic_flag_does_not_change_world_model_puzzle_serving(
    monkeypatch,
):
    from click.testing import CliRunner

    from cli.main import cli
    from config.pydantic_models import NeuralCallableConfig
    from puxle.world_model import RubiksCubeWorldModel_test

    constructed = []
    searches = []

    def fake_world_model_init(self, **kwargs):
        self.aqt_cfg = kwargs["aqt_cfg"]
        constructed.append((self, kwargs))

    class FakeNeuralHeuristic:
        def __init__(self, puzzle, **kwargs):
            self.puzzle = puzzle
            self.aqt_cfg = kwargs["aqt_cfg"]

        def distance(self, *_args):
            return 0.0

    bundle = puzzle_registry.puzzle_bundles["rubikscube_world_model_test"]
    monkeypatch.setattr(RubiksCubeWorldModel_test, "__init__", fake_world_model_init)
    monkeypatch.setattr(
        bundle,
        "heuristic_nn_configs",
        {
            "default": NeuralCallableConfig(
                callable=FakeNeuralHeuristic,
                param_path="unused.pkl",
            )
        },
    )
    monkeypatch.setattr("cli.commands.run_search_command", lambda *args: searches.append(args))

    runner = CliRunner()
    without_nn = runner.invoke(
        cli,
        ["astar-d", "-p", "rubikscube_world_model_test"],
    )
    with_nn = runner.invoke(
        cli,
        [
            "astar-d",
            "-p",
            "rubikscube_world_model_test",
            "-nn",
            "-q",
            "--quant-type",
            "int4",
        ],
    )

    assert without_nn.exit_code == 0, without_nn.output
    assert with_nn.exit_code == 0, with_nn.output
    assert [kwargs["aqt_cfg"] for _, kwargs in constructed] == ["int8", "int8"]
    assert [kwargs["init_params"] for _, kwargs in constructed] == [False, False]
    assert searches[0][0] is constructed[0][0]
    assert searches[1][0] is constructed[1][0]
    assert searches[1][7].puzzle is constructed[1][0]
    assert searches[1][7].aqt_cfg == "int4"
