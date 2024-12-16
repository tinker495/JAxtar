from abc import ABC, abstractmethod
from collections import namedtuple
from enum import Enum
from typing import Any, Dict, Type, TypeVar

import chex
import jax
import jax.numpy as jnp
from tabulate import tabulate

T = TypeVar("T")


# enum for state type
class StructuredType(Enum):
    SINGLE = 0
    BATCHED = 1
    UNSTRUCTURED = 2


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

    def get_flatten(self: "Puzzle.State"):
        batch_size = len(self)
        x = jax.tree_util.tree_map(lambda x: jnp.reshape(x, (batch_size, -1)), self)
        x_flattens, _ = jax.tree_util.tree_flatten(x)
        x_flattens = jnp.concatenate(x_flattens, axis=1)
        return x_flattens

    setattr(cls, "get_flatten", get_flatten)

    def unique_mask(self: "Puzzle.State"):
        """
        check if the dataclass is unique
        """
        size = len(self)
        flatten_x = self.get_flatten()
        unique_idx = jnp.unique(flatten_x, axis=0, size=size, return_index=True)[
            1
        ]  # shape = (size,)
        unique_masks = (
            jnp.zeros((size,), dtype=jnp.bool_).at[unique_idx].set(True)
        )  # set the unique index to True
        return unique_masks

    setattr(cls, "unique_mask", unique_mask)

    def unique_optimal(self: "Puzzle.State", key: jnp.ndarray):
        size = len(self)
        flatten_x: jnp.ndarray = self.get_flatten()  # [size, ...]

        # Get unique states and their indices
        _, inverse_idx, _ = jnp.unique(
            flatten_x, axis=0, size=size, return_inverse=True, return_counts=True
        )

        # Use size as the maximum possible number of segments
        # This is guaranteed to be large enough
        min_key_mask = jnp.zeros((size,), dtype=jnp.bool_)
        min_key_idx = jax.ops.segment_min(
            jnp.arange(size),
            inverse_idx,
            num_segments=size,  # Use size instead
            indices_are_sorted=False,
            unique_indices=True,
        )
        min_key_mask = min_key_mask.at[min_key_idx].set(True)

        return min_key_mask

    setattr(cls, "unique_optimal", unique_optimal)

    return cls


def add_string_parser(cls: Type[T], parsfunc: callable) -> Type[T]:
    """
    This function is a decorator that adds a __str__ method to
    the class that returns a string representation of the class.
    """

    def get_str(self) -> str:
        structured_type = self.structured_type

        if structured_type == StructuredType.SINGLE:
            return parsfunc(self)
        elif structured_type == StructuredType.BATCHED:
            batch_shape = self.batch_shape
            batch_len = (
                jnp.prod(jnp.array(batch_shape)) if len(batch_shape) != 1 else batch_shape[0]
            )
            results = []
            if batch_len < 20:
                for i in range(batch_len):
                    index = jnp.unravel_index(i, batch_shape)
                    current_state = jax.tree_util.tree_map(lambda x: x[index], self)
                    results.append(parsfunc(current_state))
                results.append(f"batch : {batch_shape}")
            else:
                for i in range(3):
                    index = jnp.unravel_index(i, batch_shape)
                    current_state = jax.tree_util.tree_map(lambda x: x[index], self)
                    results.append(parsfunc(current_state))
                results.append("...\n(batch : " + f"{batch_shape})")
                for i in range(batch_len - 3, batch_len):
                    index = jnp.unravel_index(i, batch_shape)
                    current_state = jax.tree_util.tree_map(lambda x: x[index], self)
                    results.append(parsfunc(current_state))
            return tabulate([results], tablefmt="plain")
        else:
            raise ValueError(f"State is not structured: {self.shape} != {self.default_shape}")

    setattr(cls, "__str__", get_str)
    return cls


def add_default(cls: Type[T], defaultfunc: callable) -> Type[T]:
    """
    This function is a decorator that adds a default dataclass to the class.
    this function for making a default dataclass with the given shape, for example, hash table of the puzzle.
    """

    def get_default(_=None) -> T:
        return defaultfunc()

    default_shape = defaultfunc().shape
    default_dim = len(default_shape[0])

    def get_default_shape(self) -> Dict[str, Any]:
        return default_shape

    def get_structured_type(self) -> StructuredType:
        if self.shape == self.default_shape:
            return StructuredType.SINGLE
        elif all(
            default_shape[k] == self.shape[k][-len(default_shape[k]) :]
            for k in range(len(cls.__annotations__.keys()))
        ):
            return StructuredType.BATCHED
        else:
            return StructuredType.UNSTRUCTURED

    def batch_shape(self) -> tuple[int, ...]:
        if self.structured_type == StructuredType.BATCHED:
            return self.shape[0][:-default_dim]
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
        print(total_length, *self[0].shape[-default_dim:])
        return jax.tree_util.tree_map(
            lambda x: jnp.reshape(x, (total_length, *x.shape[-default_dim:])), self
        )

    # add method based on default state
    setattr(cls, "default", staticmethod(jax.jit(get_default)))
    setattr(cls, "default_shape", property(get_default_shape))
    setattr(cls, "structured_type", property(get_structured_type))
    setattr(cls, "batch_shape", property(batch_shape))
    setattr(cls, "reshape", reshape)
    setattr(cls, "flatten", flatten)
    return cls


class Puzzle(ABC):
    @state_dataclass
    class State:
        """
        This class should be a dataclass that represents the state of the puzzle.
        """

        @abstractmethod
        def dtype(self):
            pass

        @abstractmethod
        def shape(self):
            pass

        @abstractmethod
        def default(_=None) -> T:
            pass

    def __init__(self):
        """
        This function should be called in the __init__ of the subclass.
        """
        super().__init__()
        self.State = add_string_parser(self.State, self.get_string_parser())
        self.State = add_default(self.State, self.get_default_gen())

        self.get_initial_state = jax.jit(self.get_initial_state)
        self.get_target_state = jax.jit(self.get_target_state)
        self.get_neighbours = jax.jit(self.get_neighbours)
        self.is_solved = jax.jit(self.is_solved)
        self.is_equal = jax.jit(self.is_equal)

    @abstractmethod
    def get_string_parser(self) -> callable:
        """
        This function should return a callable that takes a state and returns a string representation of it.
        function signature: (state: State) -> str
        """
        pass

    @abstractmethod
    def get_default_gen(self) -> callable:
        """
        This function should return a callable that takes a state and returns a shape of it.
        function signature: (state: State) -> Dict[str, Any]
        """
        pass

    @abstractmethod
    def get_initial_state(self, key=None) -> State:
        """
        This function should return a initial state.
        """
        pass

    @abstractmethod
    def get_target_state(self, key=None) -> State:
        """
        This function should return a target state.
        """
        pass

    @abstractmethod
    def get_neighbours(self, state: State, filled: bool = True) -> tuple[State, chex.Array]:
        """
        This function should return a neighbours, and the cost of the move.
        if impossible to move in a direction cost should be inf and State should be same as input state.
        """
        pass

    @abstractmethod
    def is_solved(self, state: State, target: State) -> bool:
        """
        This function should return True if the state is the target state.
        if the puzzle has multiple target states, this function should return
        True if the state is one of the target conditions.
        e.g sokoban puzzle has multiple target states. box's position should
        be the same as the target position but the player's position can be different.
        """
        pass

    def is_equal(self, state1: State, state2: State) -> bool:
        """
        This function should return True if the two states are equal.
        this functions must be all puzzle's state(dataclass) compatible, so this is not a abstract method.
        """
        tree_equal = jax.tree_util.tree_map(lambda x, y: jnp.all(x == y), state1, state2)
        return jax.tree_util.tree_reduce(jnp.logical_and, tree_equal)
