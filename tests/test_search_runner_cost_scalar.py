import jax.numpy as jnp

from cli.search_runner import _to_python_float


def test_to_python_float_handles_rank1_jax_scalar():
    value = jnp.array([20.0], dtype=jnp.float32)
    assert _to_python_float(value) == 20.0


def test_to_python_float_handles_none_and_python_scalar():
    assert _to_python_float(None) is None
    assert _to_python_float(3.5) == 3.5
