import warnings
from unittest.mock import patch

from JAxtar.bi_stars.bi_qstar import bi_qstar_builder


class _DummyPuzzle:
    action_size = 4
    State = object
    is_reversible = False


class _DummyQFunction:
    is_fixed = False


def test_bi_qstar_rejects_non_dijkstra_optimal_mode_without_explicit_unsafe_flag():
    with patch("JAxtar.bi_stars.bi_qstar.build_bi_search_result", return_value=object()):
        try:
            bi_qstar_builder(
                _DummyPuzzle(),
                _DummyQFunction(),
                batch_size=8,
                max_nodes=16,
                terminate_on_first_solution=False,
                backward_mode="value_v",
            )
        except ValueError as exc:
            assert "unsafe_allow_nonadmissible=True" in str(exc)
        else:
            raise AssertionError("Expected ValueError for unsafe non-dijkstra optimal mode")


def test_bi_qstar_allows_non_dijkstra_optimal_mode_when_unsafe_flag_is_set():
    with patch("JAxtar.bi_stars.bi_qstar.build_bi_search_result", return_value=object()):
        with patch(
            "JAxtar.bi_stars.bi_qstar._bi_qstar_loop_builder",
            return_value=(lambda *args, **kwargs: None, lambda *_: False, lambda x: x),
        ):
            with patch(
                "JAxtar.bi_stars.bi_qstar.compile_search_builder",
                side_effect=lambda fn, _puzzle, _show_compile_time, _warmup_inputs: fn,
            ):
                with warnings.catch_warnings(record=True) as caught:
                    warnings.simplefilter("always")
                    search_fn = bi_qstar_builder(
                        _DummyPuzzle(),
                        _DummyQFunction(),
                        batch_size=8,
                        max_nodes=16,
                        terminate_on_first_solution=False,
                        backward_mode="value_v",
                        unsafe_allow_nonadmissible=True,
                    )

    assert callable(search_fn)
    assert any("unsafe_allow_nonadmissible=True" in str(w.message) for w in caught)
