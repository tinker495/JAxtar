import jax.numpy as jnp

import JAxtar.id_stars.search_base as id_search_base


class _PrepareDummySearchResult:
    def __init__(self, returned_search_result):
        self.returned_search_result = returned_search_result

    def prepare_for_expansion(self, puzzle, solve_config, batch_size):
        assert batch_size == 2
        return (
            self.returned_search_result,
            jnp.array(False),
            jnp.array(False),
            jnp.array([100, 200], dtype=jnp.int32),
            jnp.array([1.0, 2.0], dtype=jnp.float32),
            jnp.array([1, 3], dtype=jnp.int32),
            jnp.array([[9, 8], [7, 6]], dtype=jnp.int32),
            jnp.array(
                [
                    [1, 2, 3, 4],
                    [5, 6, 7, 8],
                ],
                dtype=jnp.int32,
            ),
            jnp.array([True, False]),
            jnp.array([7, 8], dtype=jnp.int32),
            jnp.array([70, 80], dtype=jnp.int32),
            jnp.array([11, 22], dtype=jnp.int32),
            jnp.array([0.1, 0.2], dtype=jnp.float32),
        )


class _DedupDummySearchResult:
    def __init__(self, dedup_result, dedup_valid):
        self.dedup_result = dedup_result
        self.dedup_valid = dedup_valid

    def apply_standard_deduplication(
        self,
        flat_neighbours,
        flat_g,
        flat_valid,
        parents,
        parent_trails,
        parent_depths,
        non_backtracking_steps,
        action_size,
        flat_size,
        trail_indices,
        batch_size,
    ):
        return self.dedup_result, self.dedup_valid


def test_prepare_flat_expansion_inputs_masks_by_depth_and_trace_indices(monkeypatch):
    returned_search_result = object()
    search_result = _PrepareDummySearchResult(returned_search_result)

    def fake_build_flat_children(*args, **kwargs):
        return (
            jnp.array([1, 2, 3, 4], dtype=jnp.int32),
            jnp.array([10.0, 20.0, 30.0, 40.0], dtype=jnp.float32),
            jnp.array([2, 5, 1, 4], dtype=jnp.int32),
            jnp.array([[1, 1], [2, 2], [3, 3], [4, 4]], dtype=jnp.int32),
            jnp.array(
                [
                    [1, 2, 3, 4],
                    [5, 6, 7, 8],
                    [9, 10, 11, 12],
                    [13, 14, 15, 16],
                ],
                dtype=jnp.int32,
            ),
            jnp.array([0, 1, 0, 1], dtype=jnp.uint8),
            jnp.array([True, True, False, True]),
        )

    monkeypatch.setattr(id_search_base, "build_flat_children", fake_build_flat_children)

    out = id_search_base.prepare_flat_expansion_inputs(
        search_result=search_result,
        puzzle=object(),
        solve_config=object(),
        batch_size=2,
        action_ids=jnp.array([0, 1], dtype=jnp.uint8),
        action_size=2,
        flat_size=4,
        non_backtracking_steps=2,
        max_path_len=3,
        empty_trail_flat=jnp.zeros((4, 2), dtype=jnp.int32),
    )

    assert out[0] is returned_search_result
    assert out[15].tolist() == [True, False, False, False]
    assert out[16].tolist() == [7, -1, -1, -1]
    assert out[17].tolist() == [70, -1, -1, -1]


def test_apply_dedup_and_mask_actions_pads_invalid_histories():
    dedup_valid = jnp.array([True, False, True, False])
    dedup_result = object()
    search_result = _DedupDummySearchResult(dedup_result, dedup_valid)
    action_history = jnp.array(
        [
            [1, 2, 3],
            [4, 5, 6],
            [7, 8, 9],
            [10, 11, 12],
        ],
        dtype=jnp.int32,
    )

    out_result, out_valid, out_history = id_search_base.apply_dedup_and_mask_actions(
        search_result=search_result,
        flat_neighbours=jnp.array([1, 2, 3, 4], dtype=jnp.int32),
        flat_g=jnp.array([0.1, 0.2, 0.3, 0.4], dtype=jnp.float32),
        flat_valid=jnp.array([True, True, True, True]),
        parents=jnp.array([10, 11], dtype=jnp.int32),
        parent_trails=jnp.array([[1, 2], [3, 4]], dtype=jnp.int32),
        parent_depths=jnp.array([1, 2], dtype=jnp.int32),
        non_backtracking_steps=2,
        action_size=2,
        flat_size=4,
        trail_indices=jnp.array([0, 1], dtype=jnp.int32),
        batch_size=2,
        flat_action_history=action_history,
    )

    assert out_result is dedup_result
    assert out_valid.tolist() == [True, False, True, False]
    assert out_history[0].tolist() == [1, 2, 3]
    assert out_history[1].tolist() == [id_search_base.ACTION_PAD] * 3
    assert out_history[2].tolist() == [7, 8, 9]
    assert out_history[3].tolist() == [id_search_base.ACTION_PAD] * 3
