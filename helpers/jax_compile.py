from __future__ import annotations

import os
import re
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


# Search-jit XLA options.
# - jax 0.11.2 tightened GEMM autotuning (xla_gpu_autotune_gemm_rtol 0.1 -> 0.01), which
#   rejects the faster bf16 kernels for the heuristic MLP and slows eager A* by ~5%; keep
#   the earlier tolerance.
# - jax 0.11's default command-buffer types plus WHILE: the search while_loops run as
#   CUDA-graph device loops (conditional nodes) instead of reading every loop predicate
#   back to the host (one MemcpyD2H + host round trip per iteration). compiler_options
#   replaces the list (no "+WHILE" form). XLA ignores GPU options on other backends.
SEARCH_COMPILER_OPTIONS = {
    "xla_gpu_autotune_gemm_rtol": 0.1,
    "xla_gpu_enable_command_buffer": (
        "FUSION,CUBLAS,CUBLASLT,CUSTOM_CALL,CUDNN,DYNAMIC_SLICE_FUSION,CONDITIONAL,WHILE"
    ),
}


def _search_compiler_options() -> dict[str, Any]:
    # A per-jit option replaces xla_disable_hlo_passes instead of extending it, so carry over the
    # passes already disabled via XLA_FLAGS or by JAX (jax_compiler_enable_remat_pass=False).
    disabled = [
        name
        for value in re.findall(r"--xla_disable_hlo_passes=(\S+)", os.environ.get("XLA_FLAGS", ""))
        for name in value.split(",")
        if name
    ]
    if not jax.config.jax_compiler_enable_remat_pass:
        disabled.append("rematerialization")
    # XLA 91888df drops the async-done join of these copy fusions at command-buffer boundaries
    # (fixed upstream in openxla/xla bb767807); keep the copies synchronous.
    disabled.append("dynamic-slice-copy-fusion-async-wrapper")
    return {
        **SEARCH_COMPILER_OPTIONS,
        "xla_disable_hlo_passes": ",".join(dict.fromkeys(disabled)),
    }


def compile_search_builder(
    fn,
    puzzle,
    show_compile_time: bool = False,
    warmup_inputs=None,
):
    fn = jax.jit(fn, compiler_options=_search_compiler_options())
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
