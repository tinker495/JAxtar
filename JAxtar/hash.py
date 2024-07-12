import operator

import chex
import jax
import jax.numpy as jnp
import numpy as np
from functools import partial
from typing import Any, Dict, Type, TypeVar
from puzzle.puzzle_base import Puzzle

T = TypeVar('T')

def rotl(x, n):
    return (x << n) | (x >> (32 - n))

def to_uint32(x: chex.Array):
    bitlen = x.dtype.itemsize
    div = jnp.maximum(4 // bitlen, 1)
    pad_len = jax.lax.cond(
        x.shape[0] % div == 0,
        lambda _: 0,
        lambda _: div - x.shape[0] % div,
        None
    )
    x_padded = jnp.pad(x, (0, pad_len), mode='constant', constant_values=0)
    x_reshaped = jnp.reshape(x_padded, (-1, div))
    return jax.vmap(lambda x: jax.lax.bitcast_convert_type(x, jnp.uint32))(x_reshaped).reshape(-1)

def xxhash(x, seed):
    x = to_uint32(x)
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

capacity = int(1024 * 1024 * 2)
max_try = 128

def hashing(x, seed): # max capacity is 2 ** 24
    z = xxhash(x, seed) >> 8
    b1 = z & 255
    z = z >> 8
    b2 = z & 255
    b3 = z >> 8
    c1 = (b1 * capacity) >> 24
    c2 = (b2 * capacity) >> 16
    c3 = (b3 * capacity) >> 8
    return c1 + c2 + c3

def dataclass_hashing(x: Puzzle.State, seed: int):
    """
    x is a dataclass
    """
    def _h(x):
        # sum of hash * index for collision
        hashs = hashing(x, seed)
        return jnp.sum(hashs * jnp.arange(1, len(hashs) + 1), dtype=jnp.uint32)
    tree_hash = jax.tree_map(_h, x)
    flattened_sum_hash = sum(jax.tree_leaves(tree_hash))
    return flattened_sum_hash

def dataclass_hashing_batch(x: Puzzle.State, seed: int):
    """
    x is a dataclass
    """
    hashes = jax.vmap(lambda x: dataclass_hashing(x, seed),in_axes=0)(x)
    return hashes

@chex.dataclass
class HashTable:
    """
    Cuckoo Hash Table
    """

    seed: int
    capacity: int
    n_table: int # number of tables
    table: Puzzle.State # shape = State("args" = (capacity, cuckoo_len, ...), ...) 
    table_idx: chex.Array # shape = (capacity, ) dtype = jnp.uint4 is the index of the table in the cuckoo table.

    @staticmethod
    def make_lookup_table(statecls: Puzzle.State, seed: int, capacity: int, n_table: int = 2):
        """
        make a lookup table with the default state of the statecls
        """
        table = jax.vmap(jax.vmap(statecls.default))(jnp.zeros((capacity, n_table)))
        table_idx = jnp.zeros((capacity), dtype=jnp.uint8)
        return HashTable(seed=seed,
                        capacity=capacity,
                        n_table=n_table,
                        table=table,
                        table_idx=table_idx)

    @staticmethod
    def check(table: "HashTable", input: Puzzle.State):
        """
        find the index of the state in the table if it exists.
        if it exists return the index, cuckoo_idx and True
        if is does not exist return the unfilled index, cuckoo_idx and False
        this function could be used to check if the state is in the table or not, and insert it if it is not.
        """

        def get_new_idx(seed):
            hash_value = dataclass_hashing(input, seed)
            idx = hash_value % table.capacity
            return idx
        
        def _cond(val):
            seed, idx, table_idx, found = val
            filled_idx = table.table_idx[idx]
            in_empty = filled_idx == table_idx
            return jnp.logical_and(~found, ~in_empty)

        def _while(val):
            seed, idx, table_idx, found = val

            def get_new_idx_and_table_idx(seed, idx, table_idx):
                next_table = table_idx == table.n_table - 1
                idx, table_idx, seed = jax.lax.cond(
                    next_table,
                    lambda _: (get_new_idx(seed+1), 0, seed+1),
                    lambda _: (idx, table_idx+1, seed),
                    None
                )
                return idx, table_idx, seed
            
            def _check_equal(state1, state2):
                tree_equal = jax.tree_map(lambda x, y: jnp.all(x == y), state1, state2)
                return jax.tree_util.tree_reduce(jnp.logical_and, tree_equal)

            state = table.table[idx, table_idx]
            found = _check_equal(state, input)
            idx, table_idx, seed = jax.lax.cond(
                found,
                lambda _: (idx, table_idx, seed),
                lambda _: get_new_idx_and_table_idx(seed, idx, table_idx),
                None
            )
            return seed, idx, table_idx, found
        
        seed = table.seed
        idx = get_new_idx(seed)
        _, idx, table_idx, found = jax.lax.while_loop(
            _cond,
            _while,
            (seed, idx, 0, False)
        )
        return idx, table_idx, found
