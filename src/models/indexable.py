from typing import cast, overload

import numpy as np
from numpy.typing import ArrayLike, NDArray

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
            Sequence[float]: final state values
        """
        return self.grid[-1]

    def set_state(self, index: int, state: ArrayLike) -> None:
        """Set terminal state values to `state`

        Args:
            index (int): index of state to set
            state (Sequence[float]): final state values to set to
        """
        self.grid[index] = np.array(state)

    @overload
    def __getitem__(self, index: int) -> NDArray: ...
    @overload
    def __getitem__(self, index: tuple[int, int]) -> float: ...
    @overload
    def __getitem__(self, index: slice) -> list[NDArray]: ...
    def __getitem__(self, index: int | tuple[int, int] | slice) -> float | NDArray | list[NDArray]:
        if isinstance(index, int):
            return self.grid[index]
        if isinstance(index, tuple):
            if not isinstance(index[0], int):
                raise ValueError("Tuple Index expects int index")
            return cast(float, self.grid[index[0]][index[1]])
        if isinstance(index, slice):
            return self.grid[index]
        raise ValueError("Unsupported getitem index")


class Constant(Indexable):
    def __init__(self, value: float) -> None:
        self.value = value
        super().__init__(steps=-1)

    @overload
    def __getitem__(self, index: int) -> NDArray: ...
    @overload
    def __getitem__(self, index: tuple[int, int]) -> float: ...
    @overload
    def __getitem__(self, index: slice) -> list[NDArray]: ...
    def __getitem__(self, index: int | tuple[int, int] | slice) -> float | NDArray | list[NDArray]:
        return self.value
