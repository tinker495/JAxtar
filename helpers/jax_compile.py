from __future__ import annotations

import logging
import os
import time
from typing import Any, Callable

import jax


def compile_with_example(fn: Callable[..., Any], *args: Any) -> None:
    if _should_log():
        name = getattr(fn, "__name__", None) or getattr(fn, "__qualname__", None)
        if not name:
            name = fn.__class__.__name__
        _get_logger().info("compiling %s", name)
    lower = getattr(fn, "lower", None)
    if callable(lower):
        lower(*args).compile()
        return
    output = fn(*args)
    _block_until_ready(output)


def _block_until_ready(tree: Any) -> None:
    for leaf in jax.tree_util.tree_leaves(tree):
        if hasattr(leaf, "block_until_ready"):
            leaf.block_until_ready()
            return


def _should_log() -> bool:
    value = os.getenv("JAX_WARMUP_LOG", "").strip().lower()
    return value not in ("", "0", "false", "no")


def _get_logger() -> logging.Logger:
    logger = logging.getLogger("jax_warmup")
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter("[JAX warmup] %(message)s"))
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
        logger.propagate = False
    return logger


def compile_search_builder(
    fn,
    puzzle,
    show_compile_time: bool = False,
    warmup_inputs=None,
):
    fn = jax.jit(fn)
    if show_compile_time:
        print("initializing jit")
        start = time.time()

    if warmup_inputs is None:
        empty_solve_config = puzzle.SolveConfig.default()
        empty_states = puzzle.State.default()
        fn(empty_solve_config, empty_states)
    else:
        compile_with_example(fn, *warmup_inputs)

    if show_compile_time:
        end = time.time()
        print(f"Compile Time: {end - start:6.2f} seconds")
        print("JIT compiled\n\n")

    return fn
