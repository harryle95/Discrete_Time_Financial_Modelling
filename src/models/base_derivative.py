from src.models.base import OptionStyle, OptionType

__all__ = (
    "BaseDerivative",
    "Call",
    "Put",
    "base_derivative_factory",
)


class BaseDerivative:
    """Internal Derivative model - Used for Call/Put options"""

    def __init__(self, strike: float, expire: int, style: OptionStyle = "european") -> None:
        self.strike = strike
        self.expire = expire
        self.style = style if isinstance(style, str) else style.value

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


def base_derivative_factory(
    strike: float,
    expire: int,
    type: OptionType = "call",
    style: OptionStyle = "european",
) -> BaseDerivative:
    """Factory method to create Derivative

    Args:
        strike (float): option strike
        expire (int): expire period
        type (OptionType, optional): call/put. Defaults to "call".
        style (OptionStyle, optional): european/american. Defaults to "european".

    Returns:
        BaseDerivative: derivative class
    """
    d_type = type if isinstance(type, str) else type.value
    return (
        Call(strike=strike, expire=expire, style=style)
        if d_type == "call"
        else Put(strike=strike, expire=expire, style=style)
    )
