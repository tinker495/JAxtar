import jax.numpy as jnp
from xtructure import HashIdx

from JAxtar.stars.search_base import (
    build_action_major_parent_context,
    build_action_major_parent_layout,
    partition_and_pack_frontier_candidates,
)
from JAxtar.utils.array_ops import stable_partition_three


def test_build_action_major_parent_layout_matches_tile_semantics():
    action_size = 3
    batch_size = 4

    flat_parent_indices, flat_actions, unflatten_shape = build_action_major_parent_layout(
        action_size, batch_size
    )

    expected_parent_indices = jnp.tile(
        jnp.arange(batch_size, dtype=jnp.int32)[jnp.newaxis, :], (action_size, 1)
    ).reshape((-1,))
    expected_actions = jnp.tile(
        jnp.arange(action_size, dtype=flat_actions.dtype)[:, jnp.newaxis], (1, batch_size)
    ).reshape((-1,))

    assert unflatten_shape == (action_size, batch_size)
    assert flat_parent_indices.tolist() == expected_parent_indices.tolist()
    assert flat_actions.tolist() == expected_actions.tolist()


def test_build_action_major_parent_layout_single_action():
    flat_parent_indices, flat_actions, unflatten_shape = build_action_major_parent_layout(1, 5)

    assert unflatten_shape == (1, 5)
    assert flat_parent_indices.tolist() == [0, 1, 2, 3, 4]
    assert flat_actions.tolist() == [0, 0, 0, 0, 0]


def test_build_action_major_parent_context_matches_manual_broadcast():
    hash_idx = HashIdx(index=jnp.array([10, 20, 30, 40], dtype=jnp.uint32))
    cost = jnp.array([1.0, 2.0, 3.0, 4.0], dtype=jnp.float32)
    filled = jnp.array([True, False, True, False])
    action_size = 3
    batch_size = 4

    (
        flat_parent_hashidx,
        flat_actions,
        costs,
        filled_tiles,
        shape,
    ) = build_action_major_parent_context(
        hash_idx=hash_idx,
        cost=cost,
        filled=filled,
        action_size=action_size,
        batch_size=batch_size,
    )

    expected_parent_indices = jnp.tile(
        jnp.arange(batch_size, dtype=jnp.int32)[jnp.newaxis, :], (action_size, 1)
    ).reshape((-1,))
    expected_actions = jnp.tile(
        jnp.arange(action_size, dtype=flat_actions.dtype)[:, jnp.newaxis], (1, batch_size)
    ).reshape((-1,))
    expected_costs = jnp.broadcast_to(cost[jnp.newaxis, :], (action_size, batch_size))
    expected_filled = jnp.broadcast_to(filled[jnp.newaxis, :], (action_size, batch_size))

    assert shape == (action_size, batch_size)
    assert flat_parent_hashidx.index.tolist() == hash_idx.index[expected_parent_indices].tolist()
    assert flat_actions.tolist() == expected_actions.tolist()
    assert costs.tolist() == expected_costs.tolist()
    assert filled_tiles.tolist() == expected_filled.tolist()


def test_partition_and_pack_frontier_candidates_matches_manual_pack():
    action_size = 2
    batch_size = 3
    flat_new_states_mask = jnp.array([True, False, True, False, False, True])
    flat_process_mask = jnp.array([True, False, False, True, False, True])
    flat_neighbours = jnp.array([10, 11, 12, 13, 14, 15], dtype=jnp.int32)
    flat_costs = jnp.array([1.0, 2.0, 3.0, jnp.inf, 5.0, 6.0], dtype=jnp.float32)
    flat_hashidx = HashIdx(index=jnp.array([100, 101, 102, 103, 104, 105], dtype=jnp.uint32))

    vals, neighbours, new_states_mask, process_mask = partition_and_pack_frontier_candidates(
        flat_new_states_mask=flat_new_states_mask,
        flat_process_mask=flat_process_mask,
        flat_neighbours=flat_neighbours,
        flat_costs=flat_costs,
        flat_hashidx=flat_hashidx,
        action_size=action_size,
        batch_size=batch_size,
    )

    invperm = stable_partition_three(flat_new_states_mask, flat_process_mask)
    expected_neighbours = flat_neighbours[invperm].reshape((action_size, batch_size))
    expected_new_states_mask = flat_new_states_mask[invperm].reshape((action_size, batch_size))
    expected_process_mask = flat_process_mask[invperm].reshape((action_size, batch_size))
    expected_costs = flat_costs[invperm].reshape((action_size, batch_size))
    expected_hashidx = flat_hashidx.index[invperm].reshape((action_size, batch_size))

    assert neighbours.tolist() == expected_neighbours.tolist()
    assert new_states_mask.tolist() == expected_new_states_mask.tolist()
    assert process_mask.tolist() == expected_process_mask.tolist()
    assert vals.cost.tolist() == expected_costs.tolist()
    assert vals.hashidx.index.tolist() == expected_hashidx.tolist()
