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
    """Internal Derivative model - Used for Call/Put options"""

    def __init__(self, params: DerivativeParams) -> None:
        self.strike = params.strike
        self.expire = params.expire
        self.style = (
            params.style if isinstance(params.style, str) else params.style.value
        )
        self.params = params

    def value(self, time: int, asset: float) -> float:
        """Calculate the derivative value at t = `time`, given asset price `asset`

        Args:
            time (int): current time of assessment
            asset (float): asset price

        Returns:
            float: derivative price
        """
        if self.style == "european":
            return self.calculate_derivative(asset) if time == self.expire else 0
        return 0

    def calculate_derivative(self, asset: float) -> float:
        """Polymorphic method for determining derivative value from asset price - private method

        Args:
            asset (float): asset price

        Returns:
            float: derivative value
        """
        return NotImplemented


class Call(BaseDerivative):
    def calculate_derivative(self, asset: float) -> float:
        return max(0, asset - self.strike)


class Put(BaseDerivative):
    def calculate_derivative(self, asset: float) -> float:
        return max(0, self.strike - asset)


def base_derivative_factory(params: DerivativeParams) -> BaseDerivative:
    """Factory method for generating BaseDerivative class - either Put or Call

    Args:
        params (DerivativeParams): parameter containing type to choose from - call/put

    Returns:
        BaseDerivative: Call/Put
    """
    d_type = params.type if isinstance(params.type, str) else params.type.value
    return Call(params) if d_type == "call" else Put(params)


class Derivative(Indexable):
    """Base multistep derivative model"""

    def __init__(self, expire: int) -> None:
        self.expire = expire
        super().__init__(steps=expire)

    def compute_terminal(self, asset: Asset | None) -> Sequence[float]:
        """Polymorphic method to compute the terminal state of derivative

        Args:
            asset (Asset | None): asset model

        Returns:
            Sequence[float]: terminal derivative states
        """
        raise NotImplementedError

    def compute_derivative(self, pi: Pi) -> None:
        """Sequentially compute derivative value based on risk neutral probability
        and terminal derivative values (known or pre-computed)

        Args:
            pi (Pi): risk neutral probabilities
        """
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
        """Populate the multi-step values of derivative based on risk neutral probabilities and asset

        Call compute_terminal then compute_derivative

        Args:
            pi (Pi): risk neutral model
            asset (Asset | None): asset model
        """
        self.compute_terminal(asset)
        self.compute_derivative(pi)


class StandardDerivative(Derivative):
    """Derivative model when the termining states are not known and must be determine from asset prices"""

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
    """Derivative model when the terminal state is already known"""

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
