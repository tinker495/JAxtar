from typing import Type, TypeVar

import chex
import jax
import jax.numpy as jnp
import numpy as np
from tqdm import trange
from xtructure import StructuredType

T = TypeVar("T")


def to_uint8(input: chex.Array, active_bits: int = 1) -> chex.Array:
    if active_bits == 1:
        flatten_input = input.reshape((-1,))
        return jnp.packbits(
            flatten_input, axis=-1, bitorder="little"
        )  # input dtype: bool, output dtype: uint8
    else:
        assert jnp.issubdtype(
            input.dtype, jnp.integer
        ), f"Input must be an integer array for active_bits={active_bits} > 1, got dtype={input.dtype}"

        input_bitslen = input.dtype.itemsize * 8
        unpacked_bits = jnp.unpackbits(
            input, axis=-1, count=input_bitslen, bitorder="little"
        )  # shape: (..., input_bitslen), dtype: bool
        selected_bits = unpacked_bits[..., :active_bits]  # shape: (..., active_bits), dtype: bool
        flat_selected_bits = selected_bits.reshape(
            (-1,)
        )  # shape: (total_input_elements * active_bits,), dtype: bool
        return jnp.packbits(flat_selected_bits, axis=-1, bitorder="little")  # dtype: uint8


def from_uint8(
    packed_bytes: chex.Array, target_shape: tuple[int, ...], active_bits: int = 1
) -> chex.Array:
    assert (
        packed_bytes.dtype == jnp.uint8
    ), f"Input 'packed_bytes' must be uint8, got {packed_bytes.dtype}"

    num_target_elements = np.prod(target_shape)
    assert num_target_elements > 0, f"num_target_elements={num_target_elements} must be positive."
    assert (
        0 < active_bits <= 8
    ), f"For reconstruction, active_bits={active_bits} must be between 1 and 8 (inclusive)."

    if active_bits == 1:
        all_unpacked_bits = jnp.unpackbits(
            packed_bytes, count=num_target_elements, bitorder="little"
        )  # shape: (packed_bytes.size * 8,), dtype: bool
        return all_unpacked_bits.reshape(target_shape).astype(
            jnp.bool_
        )  # shape: target_shape, dtype: bool
    else:
        total_source_bits_needed = num_target_elements * active_bits
        all_unpacked_bits = jnp.unpackbits(
            packed_bytes, bitorder="little"
        ).flatten()  # shape: (packed_bytes.size * 8,), dtype: uint8

        assert all_unpacked_bits.size >= total_source_bits_needed, (
            f"Not enough bits ({all_unpacked_bits.size}) for num_target_elements={num_target_elements}"
            f"requiring active_bits={active_bits} each. Need total_source_bits_needed={total_source_bits_needed}."
        )

        relevant_bits = all_unpacked_bits[
            :total_source_bits_needed
        ]  # shape: (total_source_bits_needed,), dtype: uint8
        grouped_bits = relevant_bits.reshape(
            (num_target_elements, active_bits)
        )  # shape: (num_target_elements, active_bits), dtype: uint8
        reconstructed_values = jnp.packbits(
            grouped_bits, axis=-1, bitorder="little"
        )  # shape: (num_target_elements,), dtype: uint8
        return reconstructed_values.reshape(target_shape)  # shape: target_shape, dtype: uint8


def add_img_parser(cls: Type[T], imgfunc: callable) -> Type[T]:
    """
    This function is a decorator that adds a __str__ method to
    the class that returns a string representation of the class.
    """

    def get_img(self, **kwargs) -> np.ndarray:
        structured_type = self.structured_type

        if structured_type == StructuredType.SINGLE:
            return imgfunc(self, **kwargs)
        elif structured_type == StructuredType.BATCHED:
            batch_shape = self.batch_shape
            batch_len = (
                jnp.prod(jnp.array(batch_shape)) if len(batch_shape) != 1 else batch_shape[0]
            )
            results = []
            for i in trange(batch_len):
                index = jnp.unravel_index(i, batch_shape)
                current_state = jax.tree_util.tree_map(lambda x: x[index], self)
                results.append(imgfunc(current_state, **kwargs))
            results = np.stack(results, axis=0)
            return results
        else:
            raise ValueError(f"State is not structured: {self.shape} != {self.default_shape}")

    setattr(cls, "img", get_img)
    return cls


def coloring_str(string: str, color: tuple[int, int, int]) -> str:
    r, g, b = color
    return f"\x1b[38;2;{r};{g};{b}m{string}\x1b[0m"
