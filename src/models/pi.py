from typing import Any, overload

from src.helpers import calculate_down_state_prob, calculate_up_state_prob
from src.models.asset import Asset, CRRAsset
from src.models.base import BinomPiParams, CRRPiParams, TerminalPiParams

__all__ = ("Pi", "CRRPi", "BinomPi", "TerminalPi", "pi_factory", "get_pi_from_asset")


class Pi:
    """Abstract Risk neutral probability Model under binomial assumption"""

    def __init__(self, R: float, **kwargs: Any) -> None:
        self.R = R

    @property
    def p_up(self) -> float:
        """Probability of achieving the upstate"""
        return NotImplemented

    @property
    def p_down(self) -> float:
        """Probability of achieving the downstate"""
        return NotImplemented


class CRRPi(Pi):
    """Risk Neutral probability model under the CRR assumption"""

    def __init__(self, params: CRRPiParams) -> None:
        self.S_0 = params.S_0
        self.u = params.u
        self.d = params.d
        super().__init__(R=params.R)

    @property
    def p_up(self) -> float:
        return (self.R - self.d) / (self.u - self.d)

    @property
    def p_down(self) -> float:
        return (self.u - self.R) / (self.u - self.d)


class BinomPi(Pi):
    """Risk Neutral probability model under the One-Step Binomial setting"""

    def __init__(self, params: BinomPiParams) -> None:
        self.S_0 = params.S_0
        self.S_11 = params.S_11
        self.S_10 = params.S_10
        self.R = params.R

    @property
    def p_up(self) -> float:
        return calculate_up_state_prob(self.S_11, self.S_10, self.R, self.S_0)

    @property
    def p_down(self) -> float:
        return calculate_down_state_prob(self.S_11, self.S_10, self.R, self.S_0)


class TerminalPi(Pi):
    """Risk Neutral probability model when the probability values are known"""

    def __init__(self, params: TerminalPiParams) -> None:
        self._p_up = params.p_up
        self._p_down = params.p_down
        super().__init__(R=params.R)

    @property
    def p_up(self) -> float:
        return self._p_up

    @property
    def p_down(self) -> float:
        return self._p_down


@overload
def pi_factory(params: BinomPiParams) -> BinomPi: ...
@overload
def pi_factory(params: CRRPiParams) -> CRRPi: ...
@overload
def pi_factory(params: TerminalPiParams) -> TerminalPi: ...
def pi_factory(
    params: BinomPiParams | CRRPiParams | TerminalPiParams,
) -> BinomPi | CRRPi | TerminalPi:
    """Factory method that returns a matching Pi model based on params

    Args:
        params (BinomPiParams | CRRPiParams | TerminalPiParams): params

    Returns:
        BinomPi | CRRPi | TerminalPi: matching risk neutral pi model
    """
    if isinstance(params, BinomPiParams):
        return BinomPi(params)
    if isinstance(params, CRRPiParams):
        return CRRPi(params)
    if isinstance(params, TerminalPiParams):
        return TerminalPi(params)
    raise ValueError("Unexpected param type")


def get_pi_from_asset(pi_values: TerminalPiParams | None, asset: Asset | None, R: float) -> Pi:
    """Helper method to get pi model based on either pi value or asset value

    if pi values are provided, will return the TerminalPi model. Otherwise,
    will calculate pi values based on the asset model.

    Args:
        pi_values (TerminalPiParams | None): known pi value - optional
        asset (Asset | None): provided asset value - may not be known
        R (float): interest return

    Raises:
        ValueError: if both pi_values and asset are None

    Returns:
        Pi: pi model
    """
    if pi_values:
        return pi_factory(pi_values)
    if not asset:
        raise ValueError("Either asset or pi values must be provided")

    return (
        pi_factory(CRRPiParams(R=R, S_0=asset[0, 0], u=asset.u, d=asset.d))
        if isinstance(asset, CRRAsset)
        else pi_factory(
            BinomPiParams(
                R=R,
                S_0=asset[0, 0],
                S_11=asset[1, 1],
                S_10=asset[1, 0],
            )
        )
    )
