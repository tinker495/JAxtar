from typing import Type, TypeVar

import jax
import jax.numpy as jnp
import numpy as np
from tqdm import trange
from Xtructure import StructuredType

T = TypeVar("T")


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
