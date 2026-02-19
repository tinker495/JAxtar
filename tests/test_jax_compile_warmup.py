import jax.numpy as jnp

from helpers.jax_compile import _block_until_ready, jit_with_warmup


class _DummySolveConfig:
    @staticmethod
    def default():
        return jnp.array([0], dtype=jnp.int32)


class _DummyState:
    @staticmethod
    def default():
        return jnp.array([0], dtype=jnp.int32)


class _DummyPuzzle:
    SolveConfig = _DummySolveConfig
    State = _DummyState


def test_jit_with_warmup_skips_assertion_error_for_default_warmup_inputs():
    """Default warmup should be best-effort even when user code raises AssertionError."""

    def _fn(solve_config, state):
        del solve_config, state
        raise AssertionError("intentional warmup failure")

    compiled = jit_with_warmup(_fn, puzzle=_DummyPuzzle(), show_compile_time=False)
    assert callable(compiled)


class _DummySolveConfigRaises:
    @staticmethod
    def default():
        raise AssertionError("intentional solve-config warmup failure")


class _DummyPuzzleRaisesDefaults:
    SolveConfig = _DummySolveConfigRaises
    State = _DummyState


def test_jit_with_warmup_catches_default_input_construction_errors():
    """Default warmup should remain best-effort when synthetic input construction fails."""

    def _fn(solve_config, state):
        del solve_config, state
        return jnp.array([1], dtype=jnp.int32)

    compiled = jit_with_warmup(_fn, puzzle=_DummyPuzzleRaisesDefaults(), show_compile_time=False)
    assert callable(compiled)


class _BlockableLeaf:
    def __init__(self):
        self.block_count = 0

    def block_until_ready(self):
        self.block_count += 1
        return self


def test_block_until_ready_blocks_all_ready_leaves():
    """Warmup output sync should block every leaf, not just the first one."""
    first = _BlockableLeaf()
    second = _BlockableLeaf()
    tree = {"a": first, "b": [second]}

    _block_until_ready(tree)

    assert first.block_count == 1
    assert second.block_count == 1
