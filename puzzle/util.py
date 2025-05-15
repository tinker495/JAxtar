from typing import Type, TypeVar

import chex
import jax
import jax.numpy as jnp
import numpy as np
from tqdm import trange
from Xtructure import StructuredType

T = TypeVar("T")


def to_uint8(input: chex.Array, active_bits: int = 1) -> chex.Array:
    if active_bits == 1:
        # Assumes input is boolean
        return jnp.packbits(input, axis=-1, bitorder="little")
    else:
        # Assumes input is integer type
        if not jnp.issubdtype(input.dtype, jnp.integer):
            raise ValueError(
                f"For active_bits > 1 ({active_bits=}),"
                f"input must be an integer array to extract bits from, got {input.dtype=}"
            )
        if active_bits <= 0:
            raise ValueError(f"{active_bits=} must be positive.")

        input_bitslen = input.dtype.itemsize * 8
        if active_bits > input_bitslen:
            raise ValueError(
                f"{active_bits=} cannot exceed the number of bits in the input dtype ({input_bitslen=})"
            )

        # Unpack each element of the input array into its constituent bits
        # input shape (..., N), unpacked_bits shape (..., N, input_bitslen)
        unpacked_bits = jnp.unpackbits(input, axis=-1, count=input_bitslen, bitorder="little")
        # Take the LSB active_bits from each element
        # selected_bits shape (..., N, active_bits)
        selected_bits = unpacked_bits[..., :active_bits]
        # Flatten all these selected bit groups into a single 1D array of bits
        # flat_selected_bits shape (total_elements * active_bits,)
        flat_selected_bits = selected_bits.reshape((-1,))
        # Pack these bits into a new uint8 array
        return jnp.packbits(flat_selected_bits, axis=-1, bitorder="little")


def from_uint8(
    packed_bytes: chex.Array, target_shape: tuple[int, ...], active_bits: int = 1
) -> chex.Array:
    if packed_bytes.dtype != jnp.uint8:
        raise ValueError(f"Input 'packed_bytes' must be uint8, got {packed_bytes.dtype}")

    num_target_elements = np.prod(target_shape)
    if num_target_elements == 0 and not (
        len(target_shape) > 0 and 0 in target_shape
    ):  # allow empty shapes if prod is 0 due to a 0 dim
        if target_shape == ():  # Special case for scalar
            pass
        elif (
            np.prod(target_shape) != 0
        ):  # Should not happen if num_target_elements is 0 unless target_shape is like (0,)
            raise ValueError("target_shape implies non-zero elements but num_target_elements is 0.")

    if active_bits == 1:
        # Expects packed_bytes to be uint8, output is boolean
        total_bits_needed = num_target_elements
        if total_bits_needed == 0:  # Handle empty target shape
            return jnp.empty(target_shape, dtype=jnp.bool_)

        # Unpack all bits from the input uint8 array, then flatten
        all_unpacked_bits = jnp.unpackbits(packed_bytes, bitorder="little").flatten()

        if all_unpacked_bits.size < total_bits_needed:
            raise ValueError(
                f"Not enough bits in packed_bytes to fill target_shape {target_shape} ({total_bits_needed=}). "
                f"Got {all_unpacked_bits.size} bits from {packed_bytes.size} bytes."
            )
        # Select the required number of bits and reshape
        return all_unpacked_bits[:total_bits_needed].reshape(target_shape).astype(jnp.bool_)
    else:
        # Expects packed_bytes to be uint8.
        # Output is an integer array (specifically uint8 representing the value from active_bits)
        # where each element was formed from active_bits.
        if active_bits <= 0 or active_bits > 8:
            raise ValueError(
                f"For reconstruction, active_bits ({active_bits}) must be between 1 and 8 (inclusive)."
            )

        if num_target_elements == 0:  # Handle empty target shape
            return jnp.empty(
                target_shape, dtype=jnp.uint8
            )  # Return uint8 as per reconstruction logic

        total_source_bits_needed = num_target_elements * active_bits

        # Unpack all bits from the input uint8 array, then flatten
        all_unpacked_bits = jnp.unpackbits(packed_bytes, bitorder="little").flatten()

        if all_unpacked_bits.size < total_source_bits_needed:
            raise ValueError(
                f"Not enough bits in packed_bytes for {num_target_elements} elements "
                f"each requiring {active_bits} bits. Need {total_source_bits_needed}, "
                f"got {all_unpacked_bits.size} from {packed_bytes.size} bytes."
            )

        # Take the bits relevant for the target elements
        relevant_bits = all_unpacked_bits[:total_source_bits_needed]

        # Reshape into (num_target_elements, active_bits)
        # Each row now contains the bits for one original element value
        grouped_bits = relevant_bits.reshape((num_target_elements, active_bits))

        # Pack these groups of `active_bits` back into integer values.
        # jnp.packbits with axis=-1 on (N, k) array (k<=8) creates (N,) uint8 values.
        reconstructed_values = jnp.packbits(grouped_bits, axis=-1, bitorder="little")

        # Reshape to the target shape. The dtype of reconstructed_values is uint8.
        return reconstructed_values.reshape(target_shape)


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
