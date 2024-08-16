from src.helpers import (
    calculate_H0,
    calculate_H1,
    calculate_W_0_general,
    calculate_W_0_replicating,
)
from src.models import (
    CRRAssetParams,
    DerivativeParams,
    TerminalAssetParams,
    TerminalDerivativeParams,
    TerminalPiParams,
    asset_factory,
    derivative_factory,
    get_pi_from_asset,
)


class OneStepModel:
    def __init__(
        self,
        asset_params: CRRAssetParams | TerminalAssetParams,
        derivative_params: DerivativeParams | TerminalDerivativeParams,
        R: float,
        pi_values: TerminalPiParams | None = None,
    ) -> None:
        self.asset = asset_factory(asset_params)
        self.R = R
        self.pi = get_pi_from_asset(pi_values=pi_values, asset=self.asset, R=self.R)

        self.derivative = derivative_factory(derivative_params)
        self.derivative.compute_grid(self.pi, self.asset)

    @property
    def H0(self) -> float:
        return calculate_H0(
            self.asset[1, 1],
            self.asset[1, 0],
            self.derivative[1, 1],
            self.derivative[1, 0],
            self.R,
        )

    @property
    def H1(self) -> float:
        return calculate_H1(
            self.asset[1, 1],
            self.asset[1, 0],
            self.derivative[1, 1],
            self.derivative[1, 0],
        )

    @property
    def premium(self) -> float:
        return calculate_W_0_general(
            self.derivative[1, 1],
            self.derivative[1, 0],
            self.pi.p_up,
            self.pi.p_down,
            self.R,
        )

    @property
    def premium_replicating(self) -> float:
        return calculate_W_0_replicating(self.H0, self.H1, self.asset[0, 0])
