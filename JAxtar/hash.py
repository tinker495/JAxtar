import operator

import chex
import jax
import jax.numpy as jnp
import numpy as np
from typing import Any, Dict, Type, TypeVar
from puzzle.puzzle_base import Puzzle

T = TypeVar('T')

def rotl(x, n):
    return (x << n) | (x >> (32 - n))

def to_uint32(x: chex.Array):
    bitlen = x.dtype.itemsize
    div = jnp.maximum(4 // bitlen, 1)
    x = jnp.concatenate([x, jnp.zeros(div - len(x) % div, x.dtype)])
    x_reshaped = jnp.reshape(x, (-1, div))
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

def dataclass_hashing(x, seed):
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

def dataclass_hashing_batch(x, seed):
    """
    x is a dataclass
    """
    hashes = jax.vmap(lambda x: dataclass_hashing(x, seed),in_axes=0)(x)
    return hashes

@chex.dataclass
class HashTable:

    seed: int
    capacity: int
    cuckoo_len: int # number of tables
    table: Puzzle.State # shape = State("args" = (cuckoo_len, max_capacity, ...), ...) 
    filled: chex.Array # shape = (cuckoo_len, max_capacity,)

    @staticmethod
    def make_lookup_table(statecls: Puzzle.State, seed: int, capacity: int, cuckoo_len: int = 2):
        """
        dataclass is a dataclass
        """
        table = jax.vmap(jax.vmap(statecls.default))(jnp.zeros((cuckoo_len, capacity)))
        filled = jnp.zeros((cuckoo_len, capacity), dtype=jnp.bool_)
        return HashTable(seed=seed,
                        capacity=capacity,
                        cuckoo_len=cuckoo_len,
                        table=table,
                        filled=filled)

    @staticmethod
    def check(table: "HashTable", x: Puzzle.State):
        """
        find the index of the state in the table if it exists.
        if it exists return the index, cuckoo_idx and True
        if is does not exist return the unfilled index, cuckoo_idx and False
        this function could be used to check if the state is in the table or not, and insert it if it is not.
        """

        def get_new_idx(seed):
            hash_value = dataclass_hashing(x, seed)
            idx = hash_value % table.capacity
            return idx, seed+1

        def _cond(val):
            _, _, equal, not_filled = val
            return jnp.logical_or(not_filled, equal)
        
        def _check_equal(state1, state2):
            tree_equal = jax.tree_map(lambda x, y: jnp.all(x == y), state1, state2)
            return jax.tree_util.tree_reduce(jnp.logical_and, tree_equal)

        def _while_check(val):
            _, cuckoo, seed, _, _ = val
            idx, seed = jax.lax.cond(cuckoo, lambda x: get_new_idx(x), lambda x: (idx, seed), seed)
            table_state = jax.lax.cond(cuckoo, lambda idx: table.cuckoo_table[idx], lambda idx: table.table[idx], idx)
            equal = _check_equal(table_state, x)
        
        idx, _, found, _ = jax.lax.while_loop(_cond, _while_check, (0, False, table.seed, False, True))
        return idx, found

    @staticmethod
    def insert(table: "HashTable", x: Puzzle.State):
        """
        x is a dataclass
        """
        hash_value = dataclass_hashing(x, table.seed)
        idx = hash_value % table.capacity
