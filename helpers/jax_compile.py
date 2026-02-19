from __future__ import annotations

import logging
import os
import time
from typing import Any, Callable

import jax

WARMUP_COMPILE_EXCEPTIONS = (
    AssertionError,
    AttributeError,
    NotImplementedError,
    RuntimeError,
    TypeError,
    ValueError,
    jax.errors.ConcretizationTypeError,
    jax.errors.JAXIndexError,
    jax.errors.JAXTypeError,
    jax.errors.JaxRuntimeError,
    jax.errors.NonConcreteBooleanIndexError,
    jax.errors.TracerArrayConversionError,
    jax.errors.TracerBoolConversionError,
    jax.errors.TracerIntegerConversionError,
    jax.errors.UnexpectedTracerError,
)


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


def jit_with_warmup(
    fn: Callable[..., Any],
    *,
    puzzle: Any,
    show_compile_time: bool = False,
    warmup_inputs: tuple[Any, Any] | None = None,
    init_message: str | None = "initializing jit",
    elapsed_message: str | None = "Compile Time: {elapsed:6.2f} seconds",
    completion_message: str | None = "JIT compiled\n\n",
) -> Callable[..., Any]:
    """JIT-compile a search function and optionally pre-compile it with warmup inputs."""
    jitted_fn = jax.jit(fn)

    start_t: float | None = None
    if show_compile_time:
        if init_message:
            print(init_message)
        start_t = time.time()

    if warmup_inputs is None:
        try:
            empty_solve_config = puzzle.SolveConfig.default()
            empty_states = puzzle.State.default()
            compile_with_example(jitted_fn, empty_solve_config, empty_states)
        except WARMUP_COMPILE_EXCEPTIONS as exc:
            # Some search implementations cannot trace with synthetic default
            # values. In that case, defer compilation to the first real call.
            if _should_log():
                _get_logger().warning("skipping default warmup compile: %s", exc)
    else:
        compile_with_example(jitted_fn, *warmup_inputs)

    if show_compile_time:
        elapsed = time.time() - start_t if start_t is not None else 0.0
        if elapsed_message:
            print(elapsed_message.format(elapsed=elapsed))
        if completion_message:
            print(completion_message)

    return jitted_fn


def _block_until_ready(tree: Any) -> None:
    for leaf in jax.tree_util.tree_leaves(tree):
        if hasattr(leaf, "block_until_ready"):
            leaf.block_until_ready()


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
