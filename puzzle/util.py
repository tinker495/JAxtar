from collections import namedtuple
from enum import Enum
from typing import Any, Dict, Type, TypeVar

import chex
import jax
import jax.numpy as jnp
import numpy as np
from tabulate import tabulate
from tqdm import trange

MAX_PRINT_BATCH_SIZE = 4
SHOW_BATCH_SIZE = 2
T = TypeVar("T")
Puzzle = TypeVar("Puzzle")


# enum for state type
class StructuredType(Enum):
    SINGLE = 0
    BATCHED = 1
    UNSTRUCTURED = 2


def isnamedtupleinstance(x):
    t = type(x)
    b = t.__bases__
    if len(b) != 1 or b[0] != tuple:
        return False
    f = getattr(t, "_fields", None)
    if not isinstance(f, tuple):
        return False
    return all(type(n) == str for n in f)


def get_leaf_elements(data):
    """
    Extracts leaf elements from a nested tuple structure.

    Args:
        data: The nested tuple (or a single element).

    Yields:
        Leaf elements (non-tuple elements) within the nested structure.
    """
    if isnamedtupleinstance(data):
        for item in data:
            yield from get_leaf_elements(item)  # Recursively process sub-tuples
    else:
        yield data  # Yield the leaf element


def state_dataclass(cls: Type[T]) -> Type[T]:
    """
    This function is a decorator that adds some functionality to the dataclass.
    1. It adds a shape property to the class that returns the shape of each field.
    2. It adds a dtype property to the class that returns the dtype of each field.
    3. It adds a __getitem__ method to the class that returns a new instance of the
    class with each field indexed by the input index. (this is for vectorized dataclass)
    4. It adds a __len__ method to the class that returns the length of the first field.
    (this is for vectorized dataclass)
    """
    cls = chex.dataclass(cls)

    shape_tuple = namedtuple("shape", cls.__annotations__.keys())

    def get_shape(self: "Puzzle.State") -> shape_tuple:
        return shape_tuple(
            *[getattr(self, field_name).shape for field_name in cls.__annotations__.keys()]
        )

    setattr(cls, "shape", property(get_shape))

    type_tuple = namedtuple("dtype", cls.__annotations__.keys())

    def get_type(self: "Puzzle.State") -> type_tuple:
        return type_tuple(
            *[
                jnp.dtype(getattr(self, field_name).dtype)
                for field_name in cls.__annotations__.keys()
            ]
        )

    setattr(cls, "dtype", property(get_type))

    def getitem(self: "Puzzle.State", index):
        return jax.tree_util.tree_map(lambda x: x[index], self)

    setattr(cls, "__getitem__", getitem)

    def len(self: "Puzzle.State"):
        return self.shape[0][0]

    setattr(cls, "__len__", len)

    return cls


def add_string_parser(cls: Type[T], parsfunc: callable) -> Type[T]:
    """
    This function is a decorator that adds a __str__ method to
    the class that returns a string representation of the class.
    """

    def get_str(self, **kwargs) -> str:
        structured_type = self.structured_type

        if structured_type == StructuredType.SINGLE:
            return parsfunc(self, **kwargs)
        elif structured_type == StructuredType.BATCHED:
            batch_shape = self.batch_shape
            batch_len = (
                jnp.prod(jnp.array(batch_shape)) if len(batch_shape) != 1 else batch_shape[0]
            )
            results = []
            if batch_len <= MAX_PRINT_BATCH_SIZE:
                for i in range(batch_len):
                    index = jnp.unravel_index(i, batch_shape)
                    current_state = jax.tree_util.tree_map(lambda x: x[index], self)
                    kwargs_idx = {k: v[index] for k, v in kwargs.items()}
                    results.append(parsfunc(current_state, **kwargs_idx))
            else:
                for i in range(SHOW_BATCH_SIZE):
                    index = jnp.unravel_index(i, batch_shape)
                    current_state = jax.tree_util.tree_map(lambda x: x[index], self)
                    kwargs_idx = {k: v[index] for k, v in kwargs.items()}
                    results.append(parsfunc(current_state, **kwargs_idx))
                results.append("...\n(batch : " + f"{batch_shape})")
                for i in range(batch_len - SHOW_BATCH_SIZE, batch_len):
                    index = jnp.unravel_index(i, batch_shape)
                    current_state = jax.tree_util.tree_map(lambda x: x[index], self)
                    kwargs_idx = {k: v[index] for k, v in kwargs.items()}
                    results.append(parsfunc(current_state, **kwargs_idx))
            return tabulate([results], tablefmt="plain")
        else:
            raise ValueError(f"State is not structured: {self.shape} != {self.default_shape}")

    setattr(cls, "__str__", get_str)
    setattr(cls, "str", get_str)
    return cls


def add_default(cls: Type[T], defaultfunc: callable) -> Type[T]:
    """
    This function is a decorator that adds a default dataclass to the class.
    this function for making a default dataclass with the given shape, for example, hash table of the puzzle.
    """

    def get_default(_=None) -> T:
        return defaultfunc()

    setattr(cls, "default", staticmethod(jax.jit(get_default)))

    default_shape = defaultfunc().shape
    try:
        default_dim = len(default_shape[0])
    except IndexError:
        default_dim = None
        """
        if default_dim is None, it means that the default shape is not a batch.
        """
        return cls

    def get_default_shape(self) -> Dict[str, Any]:
        return default_shape

    def get_structured_type(self) -> StructuredType:
        shape = self.shape
        if shape == default_shape:
            return StructuredType.SINGLE
        elif all(
            ds == s[-max(len(ds), 1) :] or (ds == () and len(s) == 1)
            for ds, s in zip(get_leaf_elements(default_shape), get_leaf_elements(shape))
        ):
            return StructuredType.BATCHED
        else:
            return StructuredType.UNSTRUCTURED

    def batch_shape(self) -> tuple[int, ...]:
        if self.structured_type == StructuredType.BATCHED:
            shape = list(get_leaf_elements(self.shape))
            return shape[0][:-default_dim]
        else:
            raise ValueError(f"State is not structured: {self.shape} != {self.default_shape}")

    def reshape(self, new_shape: tuple[int, ...]) -> T:
        if self.structured_type == StructuredType.BATCHED:
            total_length = jnp.prod(jnp.array(self.batch_shape))
            new_total_length = jnp.prod(jnp.array(new_shape))
            batch_dim = len(self.batch_shape)
            if total_length != new_total_length:
                raise ValueError(
                    f"Total length of the state and new shape does not match: {total_length} != {new_total_length}"
                )
            return jax.tree_util.tree_map(
                lambda x: jnp.reshape(x, new_shape + x.shape[batch_dim:]), self
            )
        else:
            raise ValueError(f"State is not structured: {self.shape} != {self.default_shape}")

    def flatten(self):
        total_length = jnp.prod(jnp.array(self.batch_shape))
        return jax.tree_util.tree_map(
            lambda x: jnp.reshape(x, (total_length, *x.shape[-default_dim:])), self
        )

    # add method based on default state
    setattr(cls, "default_shape", property(get_default_shape))
    setattr(cls, "structured_type", property(get_structured_type))
    setattr(cls, "batch_shape", property(batch_shape))
    setattr(cls, "reshape", reshape)
    setattr(cls, "flatten", flatten)
    return cls


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
