import os

import jax
import jax.numpy as jnp

from helpers.jax_compile import (
    _get_logger,
    _should_log,
    compile_search_builder,
    compile_with_example,
)


def test_should_log_reads_expected_environment_values():
    original = os.environ.get("JAX_WARMUP_LOG")
    try:
        os.environ["JAX_WARMUP_LOG"] = "false"
        assert _should_log() is False

        os.environ["JAX_WARMUP_LOG"] = "1"
        assert _should_log() is True
    finally:
        if original is None:
            os.environ.pop("JAX_WARMUP_LOG", None)
        else:
            os.environ["JAX_WARMUP_LOG"] = original


def test_compile_with_example_uses_lower_path_when_available():
    calls = {"lower": 0, "compile": 0}

    class LowerResult:
        def compile(self):
            calls["compile"] += 1

    class _Callable:
        def lower(self, *_):
            calls["lower"] += 1
            return LowerResult()

        def __call__(self, *_):
            raise AssertionError("callable should never execute when lower is available")

    compile_with_example(_Callable())
    assert calls["lower"] == 1
    assert calls["compile"] == 1


def test_compile_with_example_blocks_ready_tree_nodes_without_lower():
    class _Leaf:
        def __init__(self):
            self.blocked = False

        def block_until_ready(self):
            self.blocked = True

    leaf = _Leaf()
    compile_with_example(lambda _: leaf, leaf)

    assert leaf.blocked is True


def test_compile_search_builder_builds_jitted_callable_for_puzzle_defaults():
    class _Puzzle:
        class State:
            @staticmethod
            def default():
                return jnp.array([1.0, 2.0], dtype=jnp.float32)

        class SolveConfig:
            @staticmethod
            def default():
                return jnp.array([3.0, 4.0], dtype=jnp.float32)

    def add(cfg, state):
        return cfg + state

    compiled_fn = compile_search_builder(add, _Puzzle(), show_compile_time=False)
    out = compiled_fn(_Puzzle.SolveConfig.default(), _Puzzle.State.default())

    assert (jax.device_get(out) == jnp.array([4.0, 6.0])).all()
    assert _get_logger().name == "jax_warmup"
