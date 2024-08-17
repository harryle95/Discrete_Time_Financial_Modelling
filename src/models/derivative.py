from typing import Literal

from src.models.asset import AssetModel
from src.models.base import OptionType, StateT
from src.models.base_derivative import base_derivative_factory
from src.models.indexable import Indexable
from src.models.interest import InterestRateModel
from src.models.pi import PiModel

__all__ = (
    "DerivativeModel",
    "StandardDerivative",
    "StateDerivative",
    "derivative_factory",
)


class DerivativeModel(Indexable):
    def __init__(self, pi: PiModel, R: InterestRateModel, expire: int) -> None:
        self.pi = pi
        self.R = R
        self.expire = expire
        super().__init__(expire)
        self.compute_terminal()
        self.compute_grid()

    def compute_terminal(self) -> None:
        raise NotImplementedError

    def compute_grid(self) -> None:
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


class StandardDerivative(DerivativeModel):
    def __init__(
        self,
        strike: float,
        expire: int,
        type: OptionType | Literal["call", "put"],
        asset: AssetModel,
        pi: PiModel,
        R: InterestRateModel,
    ) -> None:
        self.derivative = base_derivative_factory(strike=strike, expire=expire, type=type)
        self.asset = asset
        self.pi = pi
        super().__init__(expire=expire, pi=pi, R=R)

    def compute_terminal(self) -> None:
        self.set_state(self.expire, [self.derivative.value(self.expire, asset) for asset in self.asset[self.expire]])


class StateDerivative(DerivativeModel):
    """Derivative model when the terminal state is already known"""

    def __init__(
        self,
        states: StateT,
        expire: int,
        R: InterestRateModel,
        pi: PiModel,
    ) -> None:
        if expire not in states:
            raise ValueError("State derivative model must have known derivative value at expire period")
        self.states = states
        super().__init__(expire=expire, pi=pi, R=R)

    def compute_terminal(self) -> None:
        self.set_state(self.expire, self.states[self.expire])


def derivative_factory(
    expire: int,
    pi: PiModel,
    R: InterestRateModel,
    states: StateT | None = None,
    strike: float | None = None,
    type: OptionType | None = None,
    asset: AssetModel | None = None,
) -> StandardDerivative | StateDerivative:
    if states:
        return StateDerivative(states=states, expire=expire, R=R, pi=pi)
    if not strike or not type or not asset:
        raise ValueError("StandardDerivative model expects strike, type, asset")
    return StandardDerivative(strike=strike, expire=expire, type=type, asset=asset, pi=pi, R=R)
