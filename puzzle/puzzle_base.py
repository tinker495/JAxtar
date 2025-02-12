from abc import ABC, abstractmethod
from collections import namedtuple
from enum import Enum
from typing import Any, Dict, Type, TypeVar

import chex
import jax
import jax.numpy as jnp
import numpy as np
from tabulate import tabulate

T = TypeVar("T")

MAX_PRINT_BATCH_SIZE = 4
SHOW_BATCH_SIZE = 2


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
            if batch_len < MAX_PRINT_BATCH_SIZE:
                for i in range(batch_len):
                    index = jnp.unravel_index(i, batch_shape)
                    current_state = jax.tree_util.tree_map(lambda x: x[index], self)
                    results.append(parsfunc(current_state, **kwargs))
                results.append(f"batch : {batch_shape}")
            else:
                for i in range(SHOW_BATCH_SIZE):
                    index = jnp.unravel_index(i, batch_shape)
                    current_state = jax.tree_util.tree_map(lambda x: x[index], self)
                    results.append(parsfunc(current_state, **kwargs))
                results.append("...\n(batch : " + f"{batch_shape})")
                for i in range(batch_len - SHOW_BATCH_SIZE, batch_len):
                    index = jnp.unravel_index(i, batch_shape)
                    current_state = jax.tree_util.tree_map(lambda x: x[index], self)
                    results.append(parsfunc(current_state, **kwargs))
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
            for i in range(batch_len):
                index = jnp.unravel_index(i, batch_shape)
                current_state = jax.tree_util.tree_map(lambda x: x[index], self)
                results.append(imgfunc(current_state, **kwargs))
            return results
        else:
            raise ValueError(f"State is not structured: {self.shape} != {self.default_shape}")

    setattr(cls, "img", get_img)
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

        @abstractmethod
        def img(self) -> jnp.ndarray:
            pass

    @state_dataclass
    class SolveConfig:
        """
        This class should be a dataclass that represents the configuration for solving the puzzle.

        Generally, most puzzles terminate when the current state is equal to the target state.
        However, there are many cases where this is not true.
        Some problems have independent, fixed variables separate from the state.
        For example, in TSP, the positions of the points are independent and fixed,
        but the state consists of the path traversed and the current index.
        In TSP, we don't know the target state (optimal path), but we do know the locations of the points.
        We define these fixed parameters as the SolveConfig.

        To manage these cases, we use SolveConfig, a broader concept that encompasses the idea of a TargetState.

        # noqa: E501
        Ex) n_puzzle, rubikscube: We know the solvable configuration (TargetState) and the entire logic of the environment.
                In this case, SolveConfig only needs TargetState.
                Variables: TargetState.
        Ex) maze: The map can vary for each maze problem, so the map must be included in SolveConfig.
                A TargetState can also be defined, so it should be included.
                Variables: Map, TargetState.
        Ex) TSP: We don't know the TargetState (optimal path), but the positions of the points, which are specified for each problem, must vary.
                Therefore, SolveConfig should contain the points.
                Variables: Points.
        Ex) Dotknot: We don't know the TargetState, and there are no specific values assigned for each problem, so SolveConfig should be empty.
                Variables: None.
        """

        TargetState: "Puzzle.State"

        def dtype(self):
            pass

        def shape(self):
            pass

        def default(_=None) -> T:
            pass

    @property
    def has_target(self) -> bool:
        """
        This function should return a boolean that indicates whether the environment has a target state or not.
        """
        return "TargetState" in self.SolveConfig.__annotations__.keys()

    @property
    def only_target(self) -> bool:
        """
        This function should return a boolean that indicates whether the environment has only a target state or not.
        """
        return self.has_target and len(self.SolveConfig.__annotations__.keys()) == 1

    def __init__(self, **kwargs):
        """
        This function should be called in the __init__ of the subclass.
        """
        super().__init__()
        self.State = add_string_parser(self.State, self.get_string_parser())
        self.State = add_default(self.State, self.get_default_gen())
        self.State = add_img_parser(self.State, self.get_img_parser())
        self.SolveConfig = add_string_parser(
            self.SolveConfig, self.get_solve_config_string_parser()
        )
        self.SolveConfig = add_default(self.SolveConfig, self.get_solve_config_default_gen())
        self.SolveConfig = add_img_parser(self.SolveConfig, self.get_solve_config_img_parser())

        self.get_initial_state = jax.jit(self.get_initial_state)
        self.get_solve_config = jax.jit(self.get_solve_config)
        self.get_inits = jax.jit(self.get_inits)
        self.get_neighbours = jax.jit(self.get_neighbours)
        self.is_solved = jax.jit(self.is_solved)
        self.is_equal = jax.jit(self.is_equal)
        self.data_init()

    def data_init(self):
        """
        This function should be called in the __init__ of the subclass.
        If the puzzle need to load dataset, this function should be filled.
        """
        pass

    def get_solve_config_string_parser(self) -> callable:
        """
        This function should return a callable that takes a solve config and returns a string representation of it.
        function signature: (solve_config: SolveConfig) -> str
        """
        assert self.only_target, (
            "You should redefine this function, because this function is only for target state"
            f"has_target: {self.has_target}, only_target: {self.only_target}"
            f"SolveConfig: {self.SolveConfig.__annotations__.keys()}"
        )
        stringparser_state = self.get_string_parser()

        def stringparser(solve_config: "Puzzle.SolveConfig") -> str:
            return stringparser_state(solve_config.TargetState)

        return stringparser

    @abstractmethod
    def get_string_parser(self) -> callable:
        """
        This function should return a callable that takes a state and returns a string representation of it.
        function signature: (state: State) -> str
        """
        pass

    def get_solve_config_default_gen(self) -> SolveConfig:
        """
        This function should return a default solve config.
        """
        assert self.only_target, (
            "You should redefine this function, because this function is only for target state"
            f"has_target: {self.has_target}, only_target: {self.only_target}"
            f"SolveConfig: {self.SolveConfig.__annotations__.keys()}"
        )
        default_state = self.State.default()

        def default_gen():
            return self.SolveConfig(TargetState=default_state)

        return default_gen

    @abstractmethod
    def get_default_gen(self) -> callable:
        """
        This function should return a callable that takes a state and returns a shape of it.
        function signature: (state: State) -> Dict[str, Any]
        """
        pass

    def get_solve_config_img_parser(self) -> callable:
        """
        This function should return a callable that takes a solve config and returns a image representation of it.
        function signature: (solve_config: SolveConfig) -> jnp.ndarray
        """
        assert self.only_target, (
            "You should redefine this function, because this function is only for target state"
            f"has_target: {self.has_target}, only_target: {self.only_target}"
            f"SolveConfig: {self.SolveConfig.__annotations__.keys()}"
        )
        imgparser_state = self.get_img_parser()

        def imgparser(solve_config: "Puzzle.SolveConfig") -> jnp.ndarray:
            return imgparser_state(solve_config.TargetState)

        return imgparser

    @abstractmethod
    def get_img_parser(self) -> callable:
        """
        This function should return a callable that takes a state and returns a image representation of it.
        function signature: (state: State) -> jnp.ndarray
        """
        pass

    @abstractmethod
    def get_solve_config(self, key=None) -> SolveConfig:
        """
        This function should return a solve config.
        """
        pass

    @abstractmethod
    def get_initial_state(self, solve_config: SolveConfig, key=None) -> State:
        """
        This function should return a initial state.
        """
        pass

    def get_inits(self, key=None) -> tuple[State, SolveConfig]:
        """
        This function should return a initial state and solve config.
        """
        solve_config = self.get_solve_config(key)
        return self.get_initial_state(solve_config, key), solve_config

    @abstractmethod
    def get_neighbours(
        self, solve_config: SolveConfig, state: State, filled: bool = True
    ) -> tuple[State, chex.Array]:
        """
        This function should return a neighbours, and the cost of the move.
        if impossible to move in a direction cost should be inf and State should be same as input state.
        """
        pass

    @abstractmethod
    def is_solved(self, solve_config: SolveConfig, state: State) -> bool:
        """
        This function should return True if the state is the target state.
        if the puzzle has multiple target states, this function should return
        True if the state is one of the target conditions.
        e.g sokoban puzzle has multiple target states. box's position should
        be the same as the target position but the player's position can be different.
        """
        pass

    def action_to_string(self, action: int) -> str:
        """
        This function should return a string representation of the action.
        """
        return f"action {action}"

    def is_equal(self, state1: State, state2: State) -> bool:
        """
        This function should return True if the two states are equal.
        this functions must be all puzzle's state(dataclass) compatible, so this is not a abstract method.
        """
        tree_equal = jax.tree_util.tree_map(lambda x, y: jnp.all(x == y), state1, state2)
        return jax.tree_util.tree_reduce(jnp.logical_and, tree_equal)
