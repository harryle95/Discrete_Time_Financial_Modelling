from collections.abc import Sequence
from typing import overload

from src.models.asset import Asset
from src.models.base import DerivativeParams, TerminalDerivativeParams
from src.models.indexable import Indexable
from src.models.pi import Pi

__all__ = (
    "BaseDerivative",
    "Call",
    "Put",
    "base_derivative_factory",
    "Derivative",
    "TerminalDerivative",
    "derivative_factory",
    "StandardDerivative",
)


class BaseDerivative:
    def __init__(self, params: DerivativeParams) -> None:
        self.strike = params.strike
        self.expire = params.expire
        self.style = (
            params.style if isinstance(params.style, str) else params.style.value
        )
        self.params = params

    def value(self, time: int, asset: float) -> float:
        if self.style == "european":
            return self.calculate_derivative(asset) if time == self.expire else 0
        return 0

    def calculate_derivative(self, asset: float) -> float:
        return NotImplemented


class Call(BaseDerivative):
    def calculate_derivative(self, asset: float) -> float:
        return max(0, asset - self.strike)


class Put(BaseDerivative):
    def calculate_derivative(self, asset: float) -> float:
        return max(0, self.strike - asset)


def base_derivative_factory(params: DerivativeParams) -> BaseDerivative:
    d_type = params.type if isinstance(params.type, str) else params.type.value
    return Call(params) if d_type == "call" else Put(params)


class Derivative(Indexable):
    def __init__(self, expire: int) -> None:
        self.expire = expire
        super().__init__(steps=expire)

    def compute_terminal(self, asset: Asset | None) -> Sequence[float]:
        raise NotImplementedError

    def compute_derivative(self, pi: Pi) -> None:
        for step in range(self.expire - 1, -1, -1):
            self.grid[step] = [
                1
                / pi.R
                * (
                    pi.p_up * self.grid[step + 1][i + 1]
                    + pi.p_down * self.grid[step + 1][i]
                )
                for i in range(step + 1)
            ]

    def compute_grid(self, pi: Pi, asset: Asset | None) -> None:
        self.compute_terminal(asset)
        self.compute_derivative(pi)


class StandardDerivative(Derivative):
    def __init__(self, params: DerivativeParams) -> None:
        self.strike = params.strike
        self.derivative = base_derivative_factory(params)
        super().__init__(expire=params.expire)

    def compute_terminal(self, asset: Asset | None) -> Sequence[float]:
        if asset:
            term_value = [
                self.derivative.value(self.expire, asset)
                for asset in asset[self.expire]
            ]
            self.set_terminal(term_value)
            return term_value
        else:
            raise ValueError("Standard Derviative expects asset to be provided")


class TerminalDerivative(Derivative):
    def __init__(self, params: TerminalDerivativeParams) -> None:
        self.params = params
        super().__init__(expire=params.expire)
        self.compute_terminal(None)

    def compute_terminal(self, asset: Asset | None = None) -> Sequence[float]:
        self.set_terminal(self.params.V_T)
        return self.params.V_T


@overload
def derivative_factory(params: TerminalDerivativeParams) -> TerminalDerivative: ...
@overload
def derivative_factory(params: DerivativeParams) -> StandardDerivative: ...
def derivative_factory(
    params: DerivativeParams | TerminalDerivativeParams,
) -> StandardDerivative | TerminalDerivative:
    if isinstance(params, TerminalDerivativeParams):
        return TerminalDerivative(params)
    if isinstance(params, DerivativeParams):
        return StandardDerivative(params)
