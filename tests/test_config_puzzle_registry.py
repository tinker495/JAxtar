from config import puzzle_registry


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
    assert bundle.heuristic_nn_configs is not None
    assert bundle.q_function_nn_configs is not None
    assert any(k == "default" for k in bundle.heuristic_nn_configs)
    assert any(k == "default" for k in bundle.q_function_nn_configs)
