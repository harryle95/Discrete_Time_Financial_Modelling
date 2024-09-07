from typing import Literal

from src.models.asset import AssetModel
from src.models.base import OptionStyle, OptionType, StateT
from src.models.base_derivative import BaseDerivative
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
        type: OptionType | Literal["call", "put"] = "call",
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
        self.derivative = BaseDerivative(
            strike=strike,
            expire=expire,
            type=type,
            style=style,
        )
        super().__init__(expire)
        self.compute_terminal()
        self.compute_european() if self.style == "european" else self.compute_american()

    @property
    def premium(self) -> float:
        return self[0, 0]

    def compute_terminal(self) -> None:
        if self.states:
            self.set_state(self.expire, self.states[self.expire])
        else:
            if self.asset:
                self.set_state(
                    self.expire,
                    [self.derivative.value(self.expire, asset) for asset in self.asset[self.expire]],
                )
            else:
                raise ValueError("Either state values or asset values must be provided to compute derivative values")

    def compute_european(self) -> None:
        for n in range(self.expire - 1, -1, -1):
            self.set_state(
                n,
                [
                    1
                    / self.R[n, j]
                    * (self.pi[n, j] * self.grid[n + 1][j + 1] + (1 - self.pi[n, j]) * self.grid[n + 1][j])
                    for j in range(n + 1)
                ],
            )

    def compute_american(self) -> None:
        if self.asset:
            for n in range(self.expire - 1, -1, -1):
                self.set_state(
                    n,
                    [
                        max(
                            self.derivative.value(n, self.asset[n, j]),
                            1
                            / self.R[n, j]
                            * (self.pi[n, j] * self.grid[n + 1][j + 1] + (1 - self.pi[n, j]) * self.grid[n + 1][j]),
                        )
                        for j in range(n + 1)
                    ],
                )
        else:
            raise ValueError("American option requires asset model")
