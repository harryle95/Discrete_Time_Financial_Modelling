from typing import cast, overload

import numpy as np
from numpy.typing import ArrayLike, NDArray

from src.models.base import NumberType

__all__ = ("Indexable",)


class Indexable:
    """Mixin that provides indexing capability and a method to set terminal state"""

    def __init__(self, steps: int) -> None:
        self.steps = steps
        self.grid: list[NDArray] = [np.array([]) for _ in range(steps + 1)]

    @property
    def final(self) -> NDArray:
        """Get final state values

        Returns:
            Sequence[NumberType]: final state values
        """
        return self.grid[-1]

    def set_state(self, index: int, state: ArrayLike) -> None:
        """Set terminal state values to `state`

        Args:
            index (NumberType): index of state to set
            state (Sequence[NumberType]): final state values to set to
        """
        self.grid[index] = np.array(state)

    @overload
    def __getitem__(self, index: int) -> NDArray: ...
    @overload
    def __getitem__(self, index: tuple[int, int]) -> NumberType: ...
    @overload
    def __getitem__(self, index: slice) -> list[NDArray]: ...
    def __getitem__(self, index: int | tuple[int, int] | slice) -> NumberType | NDArray | list[NDArray]:
        if isinstance(index, NumberType):
            return self.grid[index]
        if isinstance(index, tuple):
            if not isinstance(index[0], NumberType):
                raise ValueError("Tuple Index expects NumberType index")
            return cast(NumberType, self.grid[index[0]][index[1]])
        if isinstance(index, slice):
            return self.grid[index]
        raise ValueError("Unsupported getitem index")


class Constant(Indexable):
    def __init__(self, value: NumberType) -> None:
        self.value = value
        super().__init__(steps=-1)

    @overload
    def __getitem__(self, index: int) -> NDArray: ...
    @overload
    def __getitem__(self, index: tuple[int, int]) -> NumberType: ...
    @overload
    def __getitem__(self, index: slice) -> list[NDArray]: ...
    def __getitem__(self, index: int | tuple[int, int] | slice) -> NumberType | NDArray | list[NDArray]:
        return self.value
