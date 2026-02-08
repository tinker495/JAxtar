from JAxtar.stars.search_base import (
    _build_path_reconstruction_diagnostic_message,
    _find_first_cost_drop,
)


def test_find_first_cost_drop_returns_first_violation_index():
    costs = [0.0, 1.0, 1.5, 1.4, 2.0]
    assert _find_first_cost_drop(costs) == 3


def test_find_first_cost_drop_returns_none_when_monotonic():
    costs = [0.0, 1.0, 1.0, 2.5]
    assert _find_first_cost_drop(costs) is None


def test_path_reconstruction_diagnostic_message_contains_core_fields():
    message = _build_path_reconstruction_diagnostic_message(
        loop_detected=True,
        loop_idx=17,
        corruption_detected=True,
        costs=[0.0, 3.0, 2.0],
        dists=[2.0, 1.0, 0.0],
    )
    assert "PATH_RECONSTRUCTION_DIAGNOSTIC" in message
    assert "loop_detected=True" in message
    assert "loop_idx=17" in message
    assert "corruption_detected=True" in message
    assert "first_cost_drop_idx=2" in message
    assert "prev_cost=3.0" in message
    assert "next_cost=2.0" in message
