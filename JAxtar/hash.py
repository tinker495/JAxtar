import operator

import jax
import jax.numpy as jnp
import numpy as np
from jax import lax


def rotl(x, n):
    return (x << n) | (x >> (32 - n))

def xxhash(x, seed):
    x = lax.bitcast_convert_type(x, jnp.uint32)
    prime_1 = np.uint32(0x9E3779B1)
    prime_2 = np.uint32(0x85EBCA77)
    prime_3 = np.uint32(0xC2B2AE3D)
    prime_5 = np.uint32(0x165667B1)
    acc = np.uint32(seed) + prime_5
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

@jax.jit
def batchIndex(arr, idx):
    return jax.vmap(operator.getitem)(arr, idx)

def cuckooHash(xs, seed=1):
    hash_stack = jnp.stack([hashing(xs, 0), hashing(xs, seed)], -1)
    assignment = jnp.zeros(len(xs), jnp.uint32)
    def cond_fun(val):
        _, collided, i = val
        return collided & (i < max_try)
    def body_fun(val):
        assignment, _, i = val
        hash_assigned = batchIndex(hash_stack, assignment)
        count = jnp.bincount(hash_assigned, length=capacity)
        collided = count[hash_assigned] > 1
        return jnp.where(collided, 1 - assignment, assignment), jnp.any(collided), i + 1
    
    assignment, collided, _ = lax.while_loop(cond_fun, body_fun, (assignment, True, 0))
    if collided:
        return cuckooHash(xs, seed + 1)
    return batchIndex(hash_stack, assignment), seed