from typing import overload

import numpy as np
from numpy.typing import NDArray

from src.models.asset import AssetModel
from src.models.base import OptionStyle, OptionType, StateT
from src.models.indexable import Indexable
from src.models.interest import InterestRateModel
from src.models.pi import PiModel

__all__ = ("DerivativeModel",)


class DerivativeModel(Indexable):
    def __init__(
        self,
        expire: int,
        R: InterestRateModel,
        pi: PiModel,
        strike: float = 0,
        type: OptionType = "call",
        style: OptionStyle = "european",
        asset: AssetModel | None = None,
        states: StateT | None = None,
    ) -> None:
        self.strike = strike
        self.expire = expire
        self.type = type
        self.R = R
        self.pi = pi
        self.style = style
        self.asset = asset
        self.states = states
        super().__init__(expire)
        self.compute_terminal()
        self.compute_european() if self.style == "european" else self.compute_american()

    @overload
    @staticmethod
    def value_from_next_step(R: float, pi: float, W_up: float | int, W_down: float | int) -> float: ...

    @overload
    @staticmethod
    def value_from_next_step(R: float, pi: float, W_up: NDArray, W_down: NDArray) -> NDArray: ...

    @overload
    @staticmethod
    def value_from_next_step(R: NDArray, pi: NDArray, W_up: NDArray, W_down: NDArray) -> NDArray: ...

    @staticmethod
    def value_from_next_step(
        R: float | NDArray, pi: float | NDArray, W_up: NDArray | float | int, W_down: NDArray | float | int
    ) -> NDArray | float:
        return 1 / R * (pi * W_up + (1 - pi) * W_down)

    @staticmethod
    def value_from_exercising(K: float, S: NDArray | float, type: OptionType) -> NDArray:
        type = type.value if not isinstance(type, str) else type  # noqa: A001
        value = S - K if type == "call" else K - S
        return np.maximum(value, 0)

    @property
    def premium(self) -> float:
        return self[0, 0]

    def compute_terminal(self) -> None:
        if self.states:
            self.set_state(self.expire, self.states[self.expire])
        else:
            if self.asset:
                value = self.value_from_exercising(K=self.strike, S=self.asset.final, type=self.type)
                self.set_state(self.expire, value)
            else:
                raise ValueError("Either state values or asset values must be provided to compute derivative values")

    def compute_european(self) -> None:
        for n in range(self.expire - 1, -1, -1):
            value = 1 / self.R[n] * (self.pi[n] * self.grid[n + 1][1:] + (1 - self.pi[n]) * self.grid[n + 1][: n + 1])
            self.set_state(n, value)

    def compute_american(self) -> None:
        if self.asset:
            for n in range(self.expire - 1, -1, -1):
                exercise_value = self.value_from_exercising(K=self.strike, S=self.asset[n], type=self.type)
                step_value = self.value_from_next_step(
                    R=self.R[n], pi=self.pi[n], W_up=self.grid[n + 1][1:], W_down=self.grid[n + 1][: n + 1]
                )
                self.set_state(n, np.maximum(exercise_value, step_value))
        else:
            raise ValueError("American option requires asset model")
