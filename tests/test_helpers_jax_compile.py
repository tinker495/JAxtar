from typing import NamedTuple

import jax
import jax.numpy as jnp

from helpers.jax_compile import (
    compile_search_builder,
    compile_with_example,
)
from JAxtar.id_stars.search_base import finalize_builder


def test_compile_with_example_prints_warmup_log_when_enabled(monkeypatch, capsys):
    monkeypatch.setenv("JAX_WARMUP_LOG", "1")

    compile_with_example(lambda: 1)

    assert capsys.readouterr().out == "[JAX warmup] compiling <lambda>\n"


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


def test_compile_search_builder_executes_warmup_before_return():
    executions = []

    class _Puzzle:
        class State:
            @staticmethod
            def default():
                return jnp.array([0], dtype=jnp.int32)

        class SolveConfig:
            @staticmethod
            def default():
                return jnp.array([0], dtype=jnp.int32)

    def add(cfg, state):
        jax.debug.callback(lambda value: executions.append(int(value[0])), state)
        return cfg + state

    compile_search_builder(
        add,
        _Puzzle(),
        warmup_inputs=(
            jnp.array([3], dtype=jnp.int32),
            jnp.array([7], dtype=jnp.int32),
        ),
    )

    assert executions == [7]


def test_id_search_builder_executes_warmup_before_return():
    executions = []

    class _Puzzle:
        class State:
            @staticmethod
            def default():
                return jnp.array([0], dtype=jnp.int32)

        class SolveConfig:
            @staticmethod
            def default():
                return jnp.array([0], dtype=jnp.int32)

    class _LoopState(NamedTuple):
        search_result: jax.Array

    def init_loop(cfg, state):
        jax.debug.callback(lambda value: executions.append(int(value[0])), state)
        return _LoopState(cfg + state)

    finalize_builder(
        _Puzzle(),
        init_loop=init_loop,
        cond=lambda _: jnp.array(False),
        body=lambda loop_state: loop_state,
        warmup_inputs=(
            jnp.array([3], dtype=jnp.int32),
            jnp.array([7], dtype=jnp.int32),
        ),
    )

    assert executions == [7]
