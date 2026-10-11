from __future__ import annotations

import os
import time
from typing import Any, Callable

import jax


def compile_with_example(fn: Callable[..., Any], *args: Any) -> None:
    if os.environ.get("JAX_WARMUP_LOG"):
        name = getattr(fn, "__name__", None) or getattr(fn, "__qualname__", None)
        if not name:
            name = fn.__class__.__name__
        print(f"[JAX warmup] compiling {name}")
    lower = getattr(fn, "lower", None)
    if callable(lower):
        lower(*args).compile()
        return
    output = fn(*args)
    _block_until_ready(output)


def warmup_with_example(fn: Callable[..., Any], *args: Any) -> None:
    _block_until_ready(fn(*args))


def _block_until_ready(tree: Any) -> None:
    for leaf in jax.tree_util.tree_leaves(tree):
        if hasattr(leaf, "block_until_ready"):
            leaf.block_until_ready()
            return


# jax 0.11.2 tightened XLA's GEMM autotuning tolerance (xla_gpu_autotune_gemm_rtol 0.1 ->
# 0.01), which rejects the faster bf16 kernels for the heuristic MLP and slows eager A*
# by ~5%; keep the earlier tolerance for the search.
SEARCH_COMPILER_OPTIONS = {"xla_gpu_autotune_gemm_rtol": 0.1}


def compile_search_builder(
    fn,
    puzzle,
    show_compile_time: bool = False,
    warmup_inputs=None,
):
    fn = jax.jit(fn, compiler_options=SEARCH_COMPILER_OPTIONS)
    if show_compile_time:
        print("initializing jit")
        start = time.time()

    if warmup_inputs is None:
        warmup_inputs = (puzzle.SolveConfig.default(), puzzle.State.default())
    compile_with_example(fn, *warmup_inputs)

    if show_compile_time:
        end = time.time()
        print(f"Compile Time: {end - start:6.2f} seconds")
        print("JIT compiled\n\n")

    warmup_with_example(fn, *warmup_inputs)
    return fn
