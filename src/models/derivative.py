from typing import cast, overload

import numpy as np
from numpy.typing import NDArray

from src.models.asset import AssetModel
from src.models.base import BarrierType, NumberType, OptionStyle, OptionType, StateT
from src.models.indexable import Indexable
from src.models.interest import InterestRateModel
from src.models.pi import PiModel

__all__ = ("OptionModel",)


class OptionModel(Indexable):
    @property
    def premium(self) -> NumberType:
        return NotImplemented


class Option(OptionModel):
    def __init__(
        self,
        expire: int,
        R: InterestRateModel,
        pi: PiModel,
        strike: NumberType = 0,
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
    def descendant_value(R: NumberType, pi: NumberType, W_up: NumberType, W_down: NumberType) -> NumberType: ...

    @overload
    @staticmethod
    def descendant_value(R: NumberType, pi: NumberType, W_up: NDArray, W_down: NDArray) -> NDArray: ...

    @overload
    @staticmethod
    def descendant_value(R: NDArray, pi: NDArray, W_up: NDArray, W_down: NDArray) -> NDArray: ...

    @staticmethod
    def descendant_value(
        R: NumberType | NDArray, pi: NumberType | NDArray, W_up: NDArray | NumberType, W_down: NDArray | NumberType
    ) -> NDArray | NumberType:
        return 1 / R * (pi * W_up + (1 - pi) * W_down)

    @staticmethod
    def exercised_value(K: NumberType, S: NDArray | NumberType, type: OptionType) -> NDArray:
        value = S - K if type == "call" else K - S
        return np.maximum(value, 0)

    @property
    def premium(self) -> NumberType:
        return self[0, 0]

    def compute_terminal(self) -> None:
        if self.states:
            self.set_state(self.expire, self.states[self.expire])
        else:
            if self.asset:
                value = self.exercised_value(K=self.strike, S=self.asset.final, type=self.type)
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
                exercise_value = self.exercised_value(K=self.strike, S=self.asset[n], type=self.type)
                step_value = self.descendant_value(
                    R=self.R[n], pi=self.pi[n], W_up=self.grid[n + 1][1:], W_down=self.grid[n + 1][: n + 1]
                )
                self.set_state(n, np.maximum(exercise_value, step_value))
        else:
            raise ValueError("American option requires asset model")


class BarrierOption(OptionModel):
    def __init__(
        self,
        expire: int,
        barrier: NumberType,
        R: InterestRateModel,
        pi: PiModel,
        asset: AssetModel,
        strike: NumberType,
        type: OptionType = "call",
        style: OptionStyle = "european",
        barrier_type: BarrierType = "up and in",
    ) -> None:
        self.expire = expire
        self.barrier = barrier
        self.barrier_type = barrier_type
        self.R = R
        self.pi = pi
        self.asset = asset
        self.strike = strike
        self.type = type
        self.style = style
        super().__init__(steps=expire)
        self.compute_terminal()
        self.compute_european() if self.style == "european" else self.compute_american()

    @property
    def premium(self) -> NumberType:
        value_tuple = self[0][0]
        return cast(NumberType, value_tuple[0] if self.satisfy_barrier(self.asset[0, 0]) else value_tuple[1])

    @property
    def barrier_is_in(self) -> bool:
        return "in" in self.barrier_type

    @property
    def barrier_is_up(self) -> bool:
        return "up" in self.barrier_type

    def satisfy_barrier(self, value: NumberType) -> bool:
        return value > self.barrier if self.barrier_is_up else value < self.barrier

    def exercised_value(self, K: NumberType, S: NumberType, type: OptionType) -> tuple[NumberType, NumberType]:
        value = S - K if type == "call" else K - S
        exercised_value = max([value, 0])
        if self.barrier_is_in:
            parent_value = exercised_value
            child_value = exercised_value if self.satisfy_barrier(S) else 0.0
        else:
            parent_value = 0
            child_value = 0 if self.satisfy_barrier(S) else exercised_value
        return (parent_value, child_value)

    def descendant_value(
        self,
        S: NumberType,
        W_up: tuple[NumberType, NumberType],
        W_down: tuple[NumberType, NumberType],
        pi: NumberType,
        R: NumberType,
    ) -> tuple[NumberType, NumberType]:
        child_value = 0.0
        desc_parent_value = 1 / R * (pi * W_up[0] + (1 - pi) * W_down[0])
        desc_child_value = 1 / R * (pi * W_up[1] + (1 - pi) * W_down[1])
        if self.barrier_is_in:
            parent_value = desc_parent_value
            child_value = desc_parent_value if self.satisfy_barrier(S) else desc_child_value
        else:
            parent_value = 0.0
            child_value = 0.0 if self.satisfy_barrier(S) else desc_child_value
        return (parent_value, child_value)

    def compute_terminal(self) -> None:
        if self.asset:
            value = [self.exercised_value(K=self.strike, S=S, type=self.type) for S in self.asset.final]
            self.set_state(self.expire, value)
        else:
            raise ValueError("Asset values must be provided to compute derivative values")

    def compute_european(self) -> None:
        for n in range(self.expire - 1, -1, -1):
            value = [
                self.descendant_value(
                    R=self.R[n, j],
                    pi=self.pi[n, j],
                    W_up=self.grid[n + 1][j + 1],
                    W_down=self.grid[n + 1][j],
                    S=self.asset[n, j],
                )
                for j in range(n + 1)
            ]
            self.set_state(n, value)

    def compute_american(self) -> None:
        if self.asset:
            for n in range(self.expire - 1, -1, -1):
                exercise_value = [self.exercised_value(K=self.strike, S=S, type=self.type) for S in self.asset[n]]
                step_value = [
                    self.descendant_value(
                        R=self.R[n, j],
                        pi=self.pi[n, j],
                        W_up=self.grid[n + 1][j + 1],
                        W_down=self.grid[n + 1][j],
                        S=self.asset[n, j],
                    )
                    for j in range(n + 1)
                ]
                self.set_state(n, np.maximum(exercise_value, step_value))
        else:
            raise ValueError("American option requires asset model")
