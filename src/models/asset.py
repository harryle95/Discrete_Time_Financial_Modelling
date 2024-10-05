from __future__ import annotations

import numpy as np

from src.models.base import NumberType, StateT
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


class BondAsset(AssetModel):
    def __init__(
        self,
        F: NumberType,
        steps: int,
        P: NumberType,
        c: NumberType,
    ) -> None:
        super().__init__(steps)
        self.F = F
        self.P = P
        self.c = c

    @staticmethod
    def calculate_yield(F: NumberType, P: NumberType, c: NumberType) -> NumberType:
        return c * F / P * 100

    @staticmethod
    def calcualte_price(F: NumberType, c: NumberType, r: NumberType, n: int) -> NumberType:
        return sum([c * F / (1 + r) ** t for t in range(1, n + 1)]) + F / (1 + r) ** n

    @staticmethod
    def calculate_ytm(F: NumberType, P: NumberType, c: NumberType, n: int) -> NumberType:
        C = c * F
        return (C + (F - P) / n) / ((F + P) / 2) * 100


class ZeroCouponBondAsset(BondAsset):
    def __init__(self, steps: int, P: NumberType) -> None:
        super().__init__(1, steps, P, 0)


class CRRAsset(AssetModel):
    """Asset Model under CRR assumption"""

    def __init__(self, S: NumberType, u: NumberType, d: NumberType, steps: int) -> None:
        super().__init__(steps=steps)
        self.S = S
        self.u = u
        self.d = d
        for t in range(self.steps + 1):
            pwr = np.arange(t + 1)
            state_value = self.S * np.power(self.u, pwr) * np.power(self.d, t - pwr)
            self.set_state(t, state_value)


class ForexAsset(StandardAsset):
    def __init__(self, X: StateT, F: NumberType, Rf: NumberType, steps: int) -> None:
        self.X = X
        self.F = F
        self.Rf = Rf
        super().__init__(states={k: np.array(v) * F * np.power(Rf, k) for k, v in X.items()}, steps=steps)


class CRRForexAsset(CRRAsset):
    def __init__(self, X: NumberType, u: NumberType, d: NumberType, F: NumberType, Rf: NumberType, steps: int) -> None:
        self.X = X
        self.F = F
        self.Rf = Rf
        super().__init__(S=X * F, u=u, d=d, steps=steps)


def forex_factory(
    steps: int,
    X: NumberType | StateT,
    F: NumberType,
    Rf: NumberType,
    u: NumberType | None = None,
    d: NumberType | None = None,
) -> CRRForexAsset | ForexAsset:
    if isinstance(X, dict):
        return ForexAsset(X=X, F=F, Rf=Rf, steps=steps)
    if isinstance(X, NumberType):
        if not u or not d:
            raise ValueError("CRR model expects non null u, d")
        return CRRForexAsset(X=X, u=u, d=d, F=F, Rf=Rf, steps=steps)
    raise TypeError(f"Invalid type for exchange rate: {type(X)}")


def asset_factory(
    steps: int,
    S: NumberType | StateT,
    u: NumberType | None = None,
    d: NumberType | None = None,
    P: NumberType | None = None,
) -> CRRAsset | StandardAsset | ZeroCouponBondAsset:
    """Factory method to generate asset model

    Args:
        steps (NumberType): number of steps
        S (NumberType | StateT): used for CRR model - initial price. Defaults to None.
        u (NumberType | None, optional): used for CRR model - up factor. Defaults to None.
        d (NumberType | None, optional): used for CRR model - down factor. Defaults to None.
        P (NumberType | None, optional): used for ZeroCouponBondAssetModfel - bond premium. Default to None

    Returns:
        CRRAsset | StandardAsset: model
    """
    if P is not None:
        return ZeroCouponBondAsset(steps, P)
    if isinstance(S, NumberType):
        if not u or not d:
            raise ValueError("CRR model expects non null S, u, d")
        return CRRAsset(S, u, d, steps)
    if isinstance(S, dict):
        return StandardAsset(states=S, steps=steps)
    raise TypeError(f"Invalid type for asset: {type(S)}")
