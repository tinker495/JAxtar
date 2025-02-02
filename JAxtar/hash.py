"""
Hash table implementation using Cuckoo hashing technique for efficient state storage and lookup.
This module provides functionality for hashing puzzle states and managing collisions.
"""

from functools import partial
from typing import Callable, Tuple, TypeVar

import chex
import jax
import jax.numpy as jnp

from JAxtar.annotate import HASH_POINT_DTYPE, HASH_TABLE_IDX_DTYPE, SIZE_DTYPE
from JAxtar.util import set_tree_as_condition
from puzzle.puzzle_base import Puzzle

T = TypeVar("T")
HASH_SIZE_MULTIPLIER = 2  # Multiplier for hash table size to reduce collision probability

HASH_FUNC_TYPE = Callable[[Puzzle.State, int], Tuple[jnp.uint32, jnp.ndarray]]


def rotl(x, n):
    """Rotate left operation for 32-bit integers."""
    return (x << n) | (x >> (32 - n))


@jax.jit
def xxhash(x, seed):
    """
    Implementation of xxHash algorithm for 32-bit integers.
    Args:
        x: Input value to hash
        seed: Seed value for hash function
    Returns:
        32-bit hash value
    """
    prime_1 = jnp.uint32(0x9E3779B1)
    prime_2 = jnp.uint32(0x85EBCA77)
    prime_3 = jnp.uint32(0xC2B2AE3D)
    prime_5 = jnp.uint32(0x165667B1)
    acc = jnp.uint32(seed) + prime_5
    for _ in range(4):
        lane = x & 255
        acc = acc + lane * prime_5
        acc = rotl(acc, 11) * prime_1
        x = x >> 8
    acc = acc ^ (acc >> 15)
    acc = acc * prime_2
    acc = acc ^ (acc >> 13)
    acc = acc * prime_3
    acc = acc ^ (acc >> 16)
    return acc


def hash_func_builder(x: Puzzle.State):
    """
    Build a hash function for the puzzle state dataclass.
    This function creates a JIT-compiled hash function that converts state objects to bytes
    and then to uint32 arrays for hashing.

    Args:
        x: Example puzzle state to determine the structure
    Returns:
        JIT-compiled hash function that takes a state and seed
    """

    @jax.jit
    def _to_bytes(x):
        """Convert input to byte array."""
        return jax.lax.bitcast_convert_type(x, jnp.uint8).reshape(-1)

    @jax.jit
    def _byterize(x):
        """Convert entire state tree to flattened byte array."""
        x = jax.tree_util.tree_map(_to_bytes, x)
        x, _ = jax.tree_util.tree_flatten(x)
        return jnp.concatenate(x)

    default_bytes = _byterize(x.default())
    bytes_len = default_bytes.shape[0]
    # Calculate padding needed to make byte length multiple of 4
    pad_len = jnp.where(bytes_len % 4 != 0, 4 - (bytes_len % 4), 0)

    if pad_len > 0:

        def _to_uint32(bytes):
            """Convert padded bytes to uint32 array."""
            x_padded = jnp.pad(bytes, (pad_len, 0), mode="constant", constant_values=0)
            x_reshaped = jnp.reshape(x_padded, (-1, 4))
            return jax.vmap(lambda x: jax.lax.bitcast_convert_type(x, jnp.uint32))(
                x_reshaped
            ).reshape(-1)

    else:

        def _to_uint32(bytes):
            """Convert bytes directly to uint32 array."""
            x_reshaped = jnp.reshape(bytes, (-1, 4))
            return jax.vmap(lambda x: jax.lax.bitcast_convert_type(x, jnp.uint32))(
                x_reshaped
            ).reshape(-1)

    def _h(x, seed):
        """
        Main hash function that converts state to bytes and applies xxhash.
        Returns both hash value and byte representation.
        """
        bytes = _byterize(x)
        uint32ed = _to_uint32(bytes)

        def scan_body(seed, x):
            result = xxhash(x, seed)
            return result, result

        hash_value, _ = jax.lax.scan(scan_body, seed, uint32ed)
        return hash_value, bytes

    return jax.jit(_h)


@chex.dataclass
class HashTable:
    """
    Cuckoo Hash Table Implementation

    This implementation uses multiple hash functions (specified by n_table)
    to resolve collisions. Each item can be stored in one of n_table possible positions.

    Attributes:
        seed: Initial seed for hash functions
        capacity: User-specified capacity
        _capacity: Actual internal capacity (larger than specified to handle collisions)
        size: Current number of items in table
        n_table: Number of hash functions/tables used
        table: The actual storage for states
        table_idx: Indices tracking which hash function was used for each entry
    """

    seed: int
    capacity: int
    _capacity: int
    size: int
    n_table: int  # number of tables
    table: Puzzle.State  # shape = State("args" = (capacity, cuckoo_len, ...), ...)
    table_idx: chex.Array  # shape = (capacity, ) is the index of the table in the cuckoo table.
    # hash_func: HASH_FUNC_TYPE

    @staticmethod
    def build(statecls: Puzzle.State, seed: int, capacity: int, n_table: int = 2):
        """
        Initialize a new hash table with specified parameters.

        Args:
            statecls: The puzzle state class to store
            seed: Initial seed for hash functions
            capacity: Desired capacity of the table
            n_table: Number of hash functions to use (default=2)

        Returns:
            Initialized HashTable instance
        """
        _capacity = jnp.array(
            HASH_SIZE_MULTIPLIER * capacity / n_table, SIZE_DTYPE
        )  # Internal capacity is larger to reduce collision probability
        size = SIZE_DTYPE(0)
        # Initialize table with default states
        table = jax.vmap(jax.vmap(statecls.default))(jnp.zeros((_capacity + 1, n_table)))
        table_idx = jnp.zeros((_capacity + 1), dtype=HASH_TABLE_IDX_DTYPE)
        # hash_func = hash_func_builder(statecls)
        return HashTable(
            seed=seed,
            capacity=capacity,
            _capacity=_capacity,
            size=size,
            n_table=n_table,
            table=table,
            table_idx=table_idx,
            # hash_func=hash_func,
        )

    @staticmethod
    def get_new_idx(
        hash_func: HASH_FUNC_TYPE,
        table: "HashTable",
        input: Puzzle.State,
        seed: int,
    ):
        """
        Calculate new index for input state using the hash function.

        Args:
            hash_func: Hash function to use
            table: Hash table instance
            input: State to hash
            seed: Seed for hash function

        Returns:
            Index in the table for the input state
        """
        hash_value, _ = hash_func(input, seed)
        idx = hash_value % table._capacity
        return idx

    @staticmethod
    def get_new_idx_byterized(
        hash_func: HASH_FUNC_TYPE,
        table: "HashTable",
        input: Puzzle.State,
        seed: int,
    ):
        """
        Calculate new index and return byte representation of input state.
        Similar to get_new_idx but also returns the byte representation for
        equality comparison.
        """
        hash_value, bytes = hash_func(input, seed)
        idx = hash_value % table._capacity
        return idx, bytes

    @staticmethod
    def _lookup(
        hash_func: HASH_FUNC_TYPE,
        table: "HashTable",
        input: Puzzle.State,
        idx: int,
        table_idx: int,
        seed: int,
        found: bool,
    ):
        """
        Internal lookup method that searches for a state in the table.
        Uses cuckoo hashing technique to check multiple possible locations.

        Args:
            hash_func: Hash function to use
            table: Hash table instance
            input: State to look up
            idx: Initial index to check
            table_idx: Which hash function to start with
            seed: Initial seed
            found: Whether the state has been found

        Returns:
            Tuple of (seed, idx, table_idx, found)
        """

        def _check_equal(state1, state2):
            tree_equal = jax.tree_util.tree_map(lambda x, y: jnp.all(x == y), state1, state2)
            return jax.tree_util.tree_reduce(jnp.logical_and, tree_equal)

        def _cond(val):
            seed, idx, table_idx, found = val
            filled_idx = table.table_idx[idx]
            in_empty = table_idx >= filled_idx
            return jnp.logical_and(~found, ~in_empty)

        def _while(val):
            seed, idx, table_idx, found = val

            def get_new_idx_and_table_idx(seed, idx, table_idx):
                next_table = table_idx >= (table.n_table - 1)
                seed, idx, table_idx = jax.lax.cond(
                    next_table,
                    lambda _: (
                        seed + 1,
                        HashTable.get_new_idx(hash_func, table, input, seed + 1),
                        HASH_TABLE_IDX_DTYPE(0),
                    ),
                    lambda _: (seed, idx, HASH_TABLE_IDX_DTYPE(table_idx + 1)),
                    None,
                )
                return seed, idx, table_idx

            state = table.table[idx, table_idx]
            found = _check_equal(state, input)
            seed, idx, table_idx = jax.lax.cond(
                found,
                lambda _: (seed, idx, table_idx),
                lambda _: get_new_idx_and_table_idx(seed, idx, table_idx),
                None,
            )
            return seed, idx, table_idx, found

        state = table.table[idx, table_idx]
        found = jnp.logical_or(found, _check_equal(state, input))
        update_seed, idx, table_idx, found = jax.lax.while_loop(
            _cond, _while, (seed, idx, table_idx, found)
        )
        return update_seed, idx, table_idx, found

    def lookup(table: "HashTable", hash_func: HASH_FUNC_TYPE, input: Puzzle.State):
        """
        find the index of the state in the table if it exists.
        if it exists return the index, cuckoo_idx and True
        if is does not exist return the
        """
        index = HashTable.get_new_idx(hash_func, table, input, table.seed)
        _, idx, table_idx, found = HashTable._lookup(
            hash_func, table, input, index, HASH_TABLE_IDX_DTYPE(0), table.seed, False
        )
        return idx, table_idx, found

    def insert(table: "HashTable", hash_func: HASH_FUNC_TYPE, input: Puzzle.State):
        """
        insert the state in the table
        """

        def _update_table(table: "HashTable", input: Puzzle.State, idx: int, table_idx: int):
            """
            insert the state in the table
            """
            table.table = set_tree_as_condition(table.table, idx, input, table_idx)
            table.table_idx = table.table_idx.at[idx].add(1)
            return table

        idx, table_idx, found = HashTable.lookup(hash_func, table, input)
        return (
            jax.lax.cond(
                found, lambda _: table, lambda _: _update_table(table, input, idx, table_idx), None
            ),
            ~found,
        )

    @staticmethod
    @partial(
        jax.jit,
        static_argnums=(
            0,
            2,
        ),
    )
    def make_batched(statecls: Puzzle.State, inputs: Puzzle.State, batch_size: int):
        """
        make a batched version of the inputs
        """
        count = len(inputs)
        batched = jax.tree_util.tree_map(
            lambda x, y: jnp.concatenate([x, y]),
            inputs,
            jax.vmap(statecls.default)(jnp.arange(batch_size - count)),
        )
        filled = jnp.concatenate([jnp.ones(count), jnp.zeros(batch_size - count)], dtype=jnp.bool_)
        return batched, filled

    @staticmethod
    def _parallel_insert(
        hash_func: HASH_FUNC_TYPE,
        table: "HashTable",
        inputs: Puzzle.State,
        seeds: chex.Array,
        index: chex.Array,
        updatable: chex.Array,
        batch_len: int,
    ):
        def _next_idx(seeds, _idxs, unupdateds):
            def get_new_idx_and_table_idx(seed, idx, table_idx, state):
                next_table = table_idx >= (table.n_table - 1)

                def next_table_fn(seed, table):
                    next_idx = HashTable.get_new_idx(hash_func, table, state, seed)
                    seed = seed + 1
                    return seed, next_idx, table.table_idx[next_idx].astype(jnp.uint32)

                seed, idx, table_idx = jax.lax.cond(
                    next_table,
                    next_table_fn,
                    lambda seed, _: (seed, idx, table_idx + 1),
                    seed,
                    table,
                )
                return seed, idx, table_idx

            idxs = _idxs[:, 0]
            table_idxs = _idxs[:, 1]
            seeds, idxs, table_idxs = jax.vmap(
                lambda unupdated, seed, idx, table_idx, state: jax.lax.cond(
                    unupdated,
                    lambda _: get_new_idx_and_table_idx(seed, idx, table_idx, state),
                    lambda _: (seed, idx, table_idx),
                    None,
                )
            )(unupdateds, seeds, idxs, table_idxs, inputs)
            _idxs = jnp.stack((idxs, table_idxs), axis=1)
            return seeds, _idxs

        def _cond(val):
            _, _, unupdated = val
            return jnp.any(unupdated)

        def _while(val):
            seeds, _idxs, unupdated = val
            seeds, _idxs = _next_idx(seeds, _idxs, unupdated)

            overflowed = _idxs[:, 1] >= table.n_table  # Overflowed index must be updated
            _idxs = jnp.where(updatable[:, jnp.newaxis], _idxs, jnp.full_like(_idxs, -1))
            unique_idxs = jnp.unique(_idxs, axis=0, size=batch_len, return_index=True)[
                1
            ]  # val = (unique_len, 2), unique_idxs = (unique_len,)
            not_uniques = (
                jnp.ones((batch_len,), dtype=jnp.bool_).at[unique_idxs].set(False)
            )  # set the unique index to True

            unupdated = jnp.logical_and(updatable, not_uniques)
            unupdated = jnp.logical_or(unupdated, overflowed)
            return seeds, _idxs, unupdated

        _idxs = jnp.where(updatable[:, jnp.newaxis], index, jnp.full_like(index, -1))
        unique_idxs = jnp.unique(_idxs, axis=0, size=batch_len, return_index=True)[
            1
        ]  # val = (unique_len, 2), unique_idxs = (unique_len,)
        not_uniques = (
            jnp.ones((batch_len,), dtype=jnp.bool_).at[unique_idxs].set(False)
        )  # set the unique index to True
        unupdated = jnp.logical_and(
            updatable, not_uniques
        )  # remove the unique index from the unupdated index

        seeds, index, _ = jax.lax.while_loop(_cond, _while, (seeds, _idxs, unupdated))

        idx, table_idx = index[:, 0], index[:, 1].astype(HASH_TABLE_IDX_DTYPE)
        table.table = set_tree_as_condition(table.table, updatable, inputs, idx, table_idx)
        table.table_idx = table.table_idx.at[idx].add(updatable)
        table.size += jnp.sum(updatable, dtype=SIZE_DTYPE)
        return table, idx, table_idx

    def parallel_insert(
        table: "HashTable", hash_func: HASH_FUNC_TYPE, inputs: Puzzle.State, filled: chex.Array
    ):
        """
        Parallel insertion of multiple states into the hash table.

        Args:
            hash_func: Hash function to use
            table: Hash table instance
            inputs: States to insert
            filled: Boolean array indicating which inputs are valid

        Returns:
            Tuple of (updated_table, updatable, unique_filled, idx, table_idx)

        Note:
            This implementation has a known issue with the search functionality
            after parallel insertion. This should be fixed in future versions.

        TODO: Fix search functionality after parallel insertion
        """

        # Get initial indices and byte representations
        initial_idx, bytes = jax.vmap(
            partial(HashTable.get_new_idx_byterized, hash_func), in_axes=(None, 0, None)
        )(table, inputs, table.seed)

        batch_len = filled.shape[0]

        # Find unique states to avoid duplicates
        unique_bytes_idx = jnp.unique(bytes, axis=0, size=batch_len, return_index=True)[1]
        unique = jnp.zeros((batch_len,), dtype=jnp.bool_).at[unique_bytes_idx].set(True)
        unique_filled = jnp.logical_and(filled, unique)

        # Look up each state
        seeds, idx, table_idx, found = jax.vmap(
            partial(HashTable._lookup, hash_func), in_axes=(None, 0, 0, None, None, 0)
        )(table, inputs, initial_idx, HASH_TABLE_IDX_DTYPE(0), table.seed, ~unique_filled)

        idxs = jnp.stack([idx, table_idx], axis=1, dtype=HASH_POINT_DTYPE)
        updatable = jnp.logical_and(~found, unique_filled)

        # Perform parallel insertion
        table, idx, table_idx = HashTable._parallel_insert(
            hash_func, table, inputs, seeds, idxs, updatable, batch_len
        )

        # Get final indices
        _, idx, table_idx, _ = jax.vmap(
            partial(HashTable._lookup, hash_func), in_axes=(None, 0, 0, None, None, 0)
        )(table, inputs, initial_idx, HASH_TABLE_IDX_DTYPE(0), table.seed, ~filled)

        return table, updatable, unique_filled, idx, table_idx
