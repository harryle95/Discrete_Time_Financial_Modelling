from collections.abc import Sequence
from typing import cast, overload

__all__ = ("Indexable",)


class Indexable:
    """Mixin that provides indexing capability and a method to set terminal state"""

    def __init__(self, steps: int) -> None:
        self.steps = steps
        self.grid: list[list[float]] = [[] for _ in range(steps + 1)]

    @property
    def initial(self) -> float:
        """Get initial state

        Returns:
            float: initial state value
        """
        return self.grid[0][0]

    @property
    def final(self) -> Sequence[float]:
        """Get final state values

        Returns:
            Sequence[float]: final state values
        """
        return self.grid[-1]

    def set_terminal(self, state: Sequence[float]) -> None:
        """Set terminal state values to `state`

        Args:
            state (Sequence[float]): final state values to set to
        """
        self.grid[-1] = cast(list[float], state)

    @overload
    def __getitem__(self, index: int) -> Sequence[float]: ...
    @overload
    def __getitem__(self, index: tuple[int, int]) -> float: ...
    @overload
    def __getitem__(self, index: slice) -> Sequence[Sequence[float]]: ...
    def __getitem__(self, index: int | tuple[int, int] | slice) -> float | Sequence[float] | Sequence[Sequence[float]]:
        if isinstance(index, int):
            return self.grid[index]
        if isinstance(index, tuple):
            if not isinstance(index[0], int):
                raise ValueError("Tuple Index expects int index")
            return self.grid[index[0]][index[1]]
        if isinstance(index, slice):
            return self.grid[index]
        raise ValueError("Unsupported getitem index")
