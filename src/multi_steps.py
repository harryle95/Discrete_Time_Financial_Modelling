from src.models import (
    CRRAssetParams,
    DerivativeParams,
    TerminalAssetParams,
    TerminalDerivativeParams,
    TerminalPiParams,
    asset_factory,
    derivative_factory,
    get_pi_from_asset,
    state_price_factory,
)


class MultiStepModel:
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
        self.state_price = state_price_factory(self.derivative.expire)
        self.state_price.compute_grid(self.pi)

    @property
    def premium(self) -> float:
        return sum([i * j for i, j in zip(self.state_price[-1], self.derivative[-1])])
