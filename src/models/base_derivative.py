from src.models.base import OptionStyle, OptionType

__all__ = ("BaseDerivative",)


class BaseDerivative:
    """Internal Derivative model - Used for Call/Put options"""

    def __init__(
        self,
        strike: float,
        expire: int,
        style: OptionStyle = "european",
        type: OptionType = "call",
    ) -> None:
        self.strike = strike
        self.expire = expire
        self.style = style if isinstance(style, str) else style.value
        self.type = type if isinstance(type, str) else type.value

    def value(self, time: int, asset: float) -> float:
        """Calculate the derivative value at t = `time`, given asset price `asset`

        Args:
            time (int): current time of assessment
            asset (float): asset price

        Returns:
            float: derivative price
        """
        if self.style == "european":
            if time == self.expire:
                return self.calculate_derivative(asset)
            return 0
        return self.calculate_derivative(asset)

    def calculate_derivative(self, asset: float) -> float:
        """Polymorphic method for determining derivative value from asset price

        Args:
            asset (float): asset price

        Returns:
            float: derivative value
        """
        return max(0, asset - self.strike) if self.type == "call" else max(0, self.strike - asset)
