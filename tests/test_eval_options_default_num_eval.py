from types import SimpleNamespace

from cli.options import eval_options
from config.pydantic_models import EvalOptions


@eval_options
def _capture_eval_options(**kwargs):
    return kwargs["eval_options"]


def _invoke_eval_options(
    puzzle_bundle,
    *,
    benchmark=None,
    num_eval=None,
):
    kwargs = {
        "puzzle_bundle": puzzle_bundle,
        "eval_preset": None,
        "batch_size": None,
        "show_compile_time": None,
        "max_node_size": None,
        "cost_weight": None,
        "pop_ratio": None,
        "num_eval": num_eval,
        "run_name": None,
        "use_early_stopping": None,
        "early_stop_patience": None,
        "early_stop_threshold": None,
    }
    if benchmark is not None:
        kwargs["benchmark"] = benchmark
    return _capture_eval_options(**kwargs)


def test_eval_options_defaults_to_200_without_benchmark():
    eval_opts = _invoke_eval_options(SimpleNamespace(eval_options_configs=None))
    assert eval_opts.num_eval == EvalOptions.DEFAULT_NUM_EVAL_WITHOUT_BENCHMARK


def test_eval_options_keeps_negative_default_for_benchmark():
    eval_opts = _invoke_eval_options(
        SimpleNamespace(eval_options_configs=None),
        benchmark=object(),
    )
    assert eval_opts.num_eval == -1


def test_eval_options_respects_bundle_default_when_present():
    puzzle_bundle = SimpleNamespace(eval_options_configs={"default": EvalOptions(num_eval=37)})
    eval_opts = _invoke_eval_options(puzzle_bundle)
    assert eval_opts.num_eval == 37


def test_eval_options_respects_explicit_num_eval_override():
    eval_opts = _invoke_eval_options(SimpleNamespace(eval_options_configs=None), num_eval=13)
    assert eval_opts.num_eval == 13


def test_eval_options_model_resolve_for_eval_setup_applies_default():
    eval_opts = EvalOptions(num_eval=-1).resolve_for_eval_setup(has_benchmark=False)
    assert eval_opts.num_eval == EvalOptions.DEFAULT_NUM_EVAL_WITHOUT_BENCHMARK
