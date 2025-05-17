from typing import Type, TypeVar

from xtructure import FieldDescriptor, Xtructurable, xtructure_dataclass

T = TypeVar("T")


class FieldDescriptor(FieldDescriptor):
    pass


class PuzzleState(Xtructurable):
    def packing(self, **kwargs) -> "PuzzleState":
        """
        This function should return a bit packed array that represents
        """
        pass

    def unpacking(self, **kwargs) -> "PuzzleState":
        """
        This function should return a Xtructurable object that represents the state.
        raw state is bit packed, so it is space efficient, but it is not good for observation & state transition.
        """
        pass


def state_dataclass(cls: Type[T]) -> Type[T]:
    """
    This decorator should be used to define a dataclass that represents the state.
    """

    cls = xtructure_dataclass(cls)

    if not hasattr(cls, "packing") and not hasattr(cls, "unpacking"):
        # if packing and unpacking are not implemented, return the state as is
        def packing(self) -> cls:
            return self

        setattr(cls, "packing", packing)

        def unpacking(self) -> cls:
            return self

        setattr(cls, "unpacking", unpacking)

    elif hasattr(cls, "packing") ^ hasattr(cls, "unpacking"):
        # packing and unpacking must be implemented together
        raise ValueError("State class must implement both packing and unpacking or neither")
    else:
        # packing and unpacking are implemented
        pass

    return cls
