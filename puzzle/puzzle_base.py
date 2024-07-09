import chex
import jax
import jax.numpy as jnp
from abc import ABC, abstractmethod
from typing import Any, Dict, Type, TypeVar
from collections import namedtuple

T = TypeVar('T')

def state_dataclass(cls: Type[T]) -> Type[T]:
    """
    This function is a decorator that adds some functionality to the dataclass.
    1. It adds a shape property to the class that returns the shape of each field.
    2. It adds a __getitem__ method to the class that returns a new instance of the class with each field indexed by the input index.
    """
    cls = chex.dataclass(cls)

    shape_tuple = namedtuple('shape', cls.__annotations__.keys())
    def get_shape(self) -> Dict[str, Any]:
        return shape_tuple(*[getattr(self, field_name).shape for field_name in cls.__annotations__.keys()])
    setattr(cls, 'shape', property(get_shape))

    def getitem(self, index):
        new_values = {}
        for field_name, field_value in self.__dict__.items():
            if hasattr(field_value, '__getitem__'):
                try:
                    new_values[field_name] = field_value[index]
                except IndexError:
                    new_values[field_name] = field_value
            else:
                new_values[field_name] = field_value
        return cls(**new_values)
    setattr(cls, '__getitem__', getitem)

    return cls

def add_forms(cls: Type[T], parsfunc: callable) -> Type[T]:
    """
    This function is a decorator that adds a __str__ method to the class that returns a string representation of the class.
    """

    def get_str(self) -> str:
        return parsfunc(self)
    
    setattr(cls, '__str__', get_str)
    return cls

class Puzzle(ABC):
    
    @state_dataclass
    class State:
        """
        This class should be a dataclass that represents the state of the puzzle.
        """
        pass

    def __init__(self):
        """
        This function should be called in the __init__ of the subclass.
        """
        super().__init__()
        self.State = add_forms(self.State, self.get_string_parser())

    @abstractmethod
    def get_string_parser(self) -> callable:
        """
        This function should return a callable that takes a state and returns a string representation of it.
        function signature: (state: State) -> str
        """
        pass

    @abstractmethod
    def get_initial_state(self, key = None) -> State:
        """
        This function should return a initial state.
        """
        pass

    @abstractmethod
    def get_target_state(self, key = None) -> State:
        """
        This function should return a target state.
        """
        pass

    @abstractmethod
    def get_neighbours(self, state: State) -> tuple[State, chex.Array]:
        """
        This function should return a neighbours, and the cost of the move.
        if impossible to move in a direction cost should be inf and State should be same as input state.
        """
        pass