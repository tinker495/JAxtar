import click

from cli.train_commands.dist_train_command import _resolve_eval_search_components
from config.pydantic_models import DistTrainOptions


def test_resolve_eval_metric_supports_bidirectional_heuristic_variants():
    run_label, builder_fn, extra_kwargs = _resolve_eval_search_components(
        train_options=DistTrainOptions(eval_search_metric="bi_astar"),
        search_model_name="heuristic",
    )
    assert run_label == "bi_astar"
    assert builder_fn.__name__ == "bi_astar_builder"
    assert extra_kwargs == {}

    run_label, builder_fn, extra_kwargs = _resolve_eval_search_components(
        train_options=DistTrainOptions(eval_search_metric="bi_astar_d"),
        search_model_name="heuristic",
    )
    assert run_label == "bi_astar_d"
    assert builder_fn.__name__ == "bi_astar_d_builder"
    assert extra_kwargs == {}


def test_resolve_eval_metric_supports_bidirectional_q_variant():
    run_label, builder_fn, extra_kwargs = _resolve_eval_search_components(
        train_options=DistTrainOptions(eval_search_metric="bi_qstar"),
        search_model_name="qfunction",
    )
    assert run_label == "bi_qstar"
    assert builder_fn.__name__ == "bi_qstar_builder"
    assert extra_kwargs == {}


def test_resolve_eval_metric_rejects_invalid_cross_family_metric():
    try:
        _resolve_eval_search_components(
            train_options=DistTrainOptions(eval_search_metric="bi_qstar"),
            search_model_name="heuristic",
        )
    except click.UsageError as exc:
        assert "Invalid --eval-search-metric" in str(exc)
    else:
        raise AssertionError("Expected click.UsageError for invalid heuristic eval metric")
