from collections.abc import Sequence
from typing import cast, overload

__all__ = "Indexable"


class Indexable:
    def __init__(self, steps) -> None:
        self.steps = steps
        self.grid: list[list[float]] = [[] for _ in range(steps + 1)]

    @property
    def initial(self) -> float:
        return self.grid[0][0]

    @property
    def final(self) -> Sequence[float]:
        return self.grid[-1]

    def set_terminal(self, state: Sequence[float]) -> None:
        self.grid[-1] = cast(list[float], state)

    @overload
    def __getitem__(self, index: int) -> Sequence[float]: ...
    @overload
    def __getitem__(self, index: tuple[int, int]) -> float: ...
    @overload
    def __getitem__(self, index: slice) -> Sequence[Sequence[float]]: ...
    def __getitem__(
        self, index: int | tuple[int, int] | slice
    ) -> float | Sequence[float] | Sequence[Sequence[float]]:
        if isinstance(index, int):
            return self.grid[index]
        if isinstance(index, tuple):
            if not isinstance(index[0], int):
                raise ValueError("Tuple Index expects int index")
            return self.grid[index[0]][index[1]]
        if isinstance(index, slice):
            return self.grid[index]
