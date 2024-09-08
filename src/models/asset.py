import numpy as np

from src.models.base import StateT
from src.models.indexable import Indexable

__all__ = ("StandardAsset", "CRRAsset", "asset_factory")


class AssetModel(Indexable): ...


class StandardAsset(AssetModel):
    """Base Asset Model"""

    def __init__(self, states: StateT, steps: int) -> None:
        super().__init__(steps)
        if 0 not in states:
            raise ValueError("Asset Model requires current asset value at t=0")
        for time, state in states.items():
            self.set_state(time, state)


class CRRAsset(AssetModel):
    """Asset Model under CRR assumption"""

    def __init__(self, S: float | int, u: float, d: float, steps: int) -> None:
        super().__init__(steps=steps)
        self.S = S
        self.u = u
        self.d = d
        for t in range(self.steps + 1):
            pwr = np.arange(t + 1)
            state_value = self.S * np.power(self.u, pwr) * np.power(self.d, t - pwr)
            self.set_state(t, state_value)


class ForExAsset(StandardAsset):
    def __init__(self, X: StateT, F: float | int, Rf: float, steps: int) -> None:
        self.X = X
        self.F = F
        self.Rf = Rf
        super().__init__(states={k: np.array(v) * F * np.power(Rf, k) for k, v in X.items()}, steps=steps)


class CRRForExAsset(CRRAsset):
    def __init__(self, X: float | int, u: float, d: float, F: float | int, Rf: float, steps: int) -> None:
        self.X = X
        self.F = F
        self.Rf = Rf
        super().__init__(S=X * F, u=u, d=d, steps=steps)


def forex_factory(
    steps: int,
    X: int | float | StateT,
    F: float,
    Rf: float,
    u: float | None = None,
    d: float | None = None,
) -> CRRForExAsset | ForExAsset:
    if isinstance(X, dict):
        return ForExAsset(X=X, F=F, Rf=Rf, steps=steps)
    if isinstance(X, int | float):
        if not u or not d:
            raise ValueError("CRR model expects non null u, d")
        return CRRForExAsset(X=X, u=u, d=d, F=F, Rf=Rf, steps=steps)
    raise TypeError(f"Invalid type for exchange rate: {type(X)}")


def asset_factory(
    steps: int,
    S: int | float | StateT,
    u: float | None = None,
    d: float | None = None,
) -> CRRAsset | StandardAsset:
    """Factory method to generate asset model

    Args:
        steps (int): number of steps
        S (float | StateT): used for CRR model - initial price. Defaults to None.
        u (float | None, optional): used for CRR model - up factor. Defaults to None.
        d (float | None, optional): used for CRR model - down factor. Defaults to None.

    Returns:
        CRRAsset | StandardAsset: model
    """
    if isinstance(S, float | int):
        if not u or not d:
            raise ValueError("CRR model expects non null S, u, d")
        return CRRAsset(S, u, d, steps)
    if isinstance(S, dict):
        return StandardAsset(states=S, steps=steps)
    raise TypeError(f"Invalid type for asset: {type(S)}")
