import chex
import jax.numpy as jnp
import pytest
from xtructure import base_dataclass

from JAxtar.stars.search_base import insert_priority_queue_batches


@base_dataclass
class _DummyPriorityQueue:
    inserted_keys: chex.Array
    inserted_vals: chex.Array
    call_count: chex.Array

    def insert(self, key_row: chex.Array, val_row: chex.Array) -> "_DummyPriorityQueue":
        idx = self.call_count.astype(jnp.int32)
        return _DummyPriorityQueue(
            inserted_keys=self.inserted_keys.at[idx].set(key_row),
            inserted_vals=self.inserted_vals.at[idx].set(val_row),
            call_count=idx + 1,
        )


@base_dataclass
class _DummySearchResult:
    priority_queue: _DummyPriorityQueue


def _build_dummy_search_result(max_calls: int, row_width: int) -> _DummySearchResult:
    pq = _DummyPriorityQueue(
        inserted_keys=jnp.full((max_calls, row_width), -1.0, dtype=jnp.float32),
        inserted_vals=jnp.full((max_calls, row_width), -1, dtype=jnp.int32),
        call_count=jnp.array(0, dtype=jnp.int32),
    )
    return _DummySearchResult(priority_queue=pq)


def test_insert_priority_queue_batches_skips_rows_without_candidates():
    keys = jnp.array(
        [
            [1.0, 1.1],
            [2.0, 2.1],
            [3.0, 3.1],
        ],
        dtype=jnp.float32,
    )
    vals = jnp.array(
        [
            [10, 11],
            [20, 21],
            [30, 31],
        ],
        dtype=jnp.int32,
    )
    masks = jnp.array(
        [
            [True, False],
            [False, False],
            [True, True],
        ]
    )

    sr = _build_dummy_search_result(max_calls=3, row_width=2)
    out = insert_priority_queue_batches(sr, keys, vals, masks)

    assert int(out.priority_queue.call_count) == 2
    assert out.priority_queue.inserted_keys[0].tolist() == pytest.approx([1.0, 1.1])
    assert out.priority_queue.inserted_keys[1].tolist() == pytest.approx([3.0, 3.1])
    assert out.priority_queue.inserted_keys[2].tolist() == pytest.approx([-1.0, -1.0])
    assert out.priority_queue.inserted_vals[0].tolist() == [10, 11]
    assert out.priority_queue.inserted_vals[1].tolist() == [30, 31]
    assert out.priority_queue.inserted_vals[2].tolist() == [-1, -1]


def test_insert_priority_queue_batches_noop_when_all_rows_masked_out():
    keys = jnp.array(
        [
            [1.0, 1.1],
            [2.0, 2.1],
        ],
        dtype=jnp.float32,
    )
    vals = jnp.array(
        [
            [10, 11],
            [20, 21],
        ],
        dtype=jnp.int32,
    )
    masks = jnp.array(
        [
            [False, False],
            [False, False],
        ]
    )

    sr = _build_dummy_search_result(max_calls=2, row_width=2)
    out = insert_priority_queue_batches(sr, keys, vals, masks)

    assert int(out.priority_queue.call_count) == 0
    assert jnp.allclose(out.priority_queue.inserted_keys, jnp.array([[-1.0, -1.0], [-1.0, -1.0]]))
    assert out.priority_queue.inserted_vals.tolist() == [[-1, -1], [-1, -1]]
