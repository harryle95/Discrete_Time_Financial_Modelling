from __future__ import annotations

from typing import Any, cast, overload

from src.helpers import calculate_H0, calculate_H1, calculate_W_0_general, calculate_W_0_replicating
from src.models import (
    AssetModel,
    InterestRateModel,
    OptionModel,
    OptionStyle,
    OptionType,
    PiModel,
    StatePriceModel,
    StateT,
    asset_factory,
    interest_factory,
    pi_factory,
)
from src.models.asset import forex_factory

__all__ = ("AssetOptionSolver", "OneStepAssetOptionSolver")


class OptionSolver:
    def __init__(
        self,
        expire: int,
        asset: AssetModel,
        R: InterestRateModel,
        pi: PiModel,
        derivative: OptionModel,
    ) -> None:
        self.expire = expire
        self.asset = asset
        self.R = R
        self.pi = pi
        self.derivative = derivative

    @property
    def state_price(self) -> StatePriceModel:
        self._state_price = StatePriceModel(pi=self.pi, R=self.R, steps=self.expire)
        return self._state_price

    @property
    def premium_state_price(self) -> float:
        return cast(float, sum([i * j for i, j in zip(self.state_price[-1], self.derivative[-1], strict=True)]))


class OneStepOptionSolver(OptionSolver):
    def __init__(
        self,
        asset: AssetModel,
        R: InterestRateModel,
        pi: PiModel,
        derivative: OptionModel,
    ) -> None:
        self.expire = 1
        self.asset = asset
        self.R = R
        self.pi = pi
        self.derivative = derivative

    @property
    def H0(self) -> float:
        return calculate_H0(
            self.asset[1, 1],
            self.asset[1, 0],
            self.derivative[1, 1],
            self.derivative[1, 0],
            self.R[0, 0],
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
            self.pi[0, 0],
            1 - self.pi[0, 0],
            self.R[0, 0],
        )

    @property
    def premium_replicating(self) -> float:
        return calculate_W_0_replicating(self.H0, self.H1, self.asset[0, 0])


class AssetOptionSolver(OptionSolver):
    @overload
    def __init__(
        self,
        *,
        expire: int,
        S: float | int,
        u: float,
        d: float,
        W: StateT | None = None,
        K: float = 0,
        type: OptionType = "call",
        style: OptionStyle = "european",
        R: float | StateT = 1.0,
        **kwargs: Any,
    ) -> None: ...
    @overload
    def __init__(
        self,
        *,
        expire: int,
        S: StateT,
        pi: float | StateT,
        W: StateT | None = None,
        K: float = 0,
        type: OptionType = "call",
        style: OptionStyle = "european",
        R: float | StateT = 1.0,
        **kwargs: Any,
    ) -> None: ...

    def __init__(
        self,
        *,
        expire: int,
        S: float | int | StateT,
        u: float | None = None,
        d: float | None = None,
        W: StateT | None = None,
        K: float = 0,
        type: OptionType = "call",
        style: OptionStyle = "european",
        R: float | StateT = 1.0,
        pi: float | StateT | None = None,
        **kwargs: Any,
    ) -> None:
        _asset = asset_factory(steps=expire, S=S, u=u, d=d)
        _R = interest_factory(R=R, steps=expire)
        _pi = pi_factory(pi=pi, steps=expire, asset=_asset, R=_R)
        _derivative = OptionModel(
            expire=expire,
            pi=_pi,
            R=_R,
            states=W,
            strike=K,
            type=type,
            asset=_asset,
            style=style,
        )
        super().__init__(expire=expire, asset=_asset, R=_R, pi=_pi, derivative=_derivative)


class OneStepAssetOptionSolver(OneStepOptionSolver):
    @overload
    def __init__(
        self,
        *,
        S: float | int,
        u: float,
        d: float,
        K: float = 0,
        type: OptionType = "call",
        style: OptionStyle = "european",
        R: float | StateT = 1.0,
        pi: float | None = None,
        **kwargs: Any,
    ) -> None: ...
    @overload
    def __init__(
        self,
        *,
        S: StateT,
        W: StateT | None = None,
        R: float | StateT = 1.0,
        pi: float | None = None,
        **kwargs: Any,
    ) -> None: ...

    def __init__(
        self,
        *,
        S: float | StateT | int,
        u: float | None = None,
        d: float | None = None,
        W: StateT | None = None,
        K: float = 0,
        type: OptionType = "call",
        style: OptionStyle = "european",
        R: float | StateT = 1.0,
        pi: float | None = None,
        **kwargs: Any,
    ) -> None:
        _asset = asset_factory(steps=1, S=S, u=u, d=d)
        _R = interest_factory(R=R, steps=1)
        _pi = pi_factory(pi=pi, steps=1, asset=_asset, R=_R)
        _derivative = OptionModel(
            expire=1,
            pi=_pi,
            R=_R,
            states=W,
            strike=K,
            type=type,
            asset=_asset,
            style=style,
        )
        super().__init__(asset=_asset, R=_R, pi=_pi, derivative=_derivative)


class ForExOptionSolver(OptionSolver):
    def __init__(
        self,
        *,
        expire: int,
        X: float | int | StateT,
        F: float | int,
        Rf: float,
        Rd: float,
        k: float,
        u: float | None = None,
        d: float | None = None,
        pi: float | None = None,
        type: OptionType = "call",
        style: OptionStyle = "european",
        W: StateT | None = None,
        **kwargs: Any,
    ) -> None:
        self.Rf = Rf
        self.k = k
        self.F = F
        _asset = forex_factory(steps=expire, X=X, F=F, Rf=Rf, u=u, d=d)
        _R = interest_factory(R=Rd, steps=expire)
        _pi = pi_factory(pi=pi, steps=expire, asset=_asset, R=_R)
        _derivative = OptionModel(
            expire=expire,
            pi=_pi,
            R=_R,
            states=W,
            strike=k * F,
            type=type,
            asset=_asset,
            style=style,
        )
        super().__init__(expire=expire, asset=_asset, R=_R, pi=_pi, derivative=_derivative)


class OneStepForExOptionSolver(OneStepOptionSolver):
    def __init__(
        self,
        *,
        X: float | int | StateT,
        F: float | int,
        Rf: float,
        Rd: float,
        k: float,
        u: float | None = None,
        d: float | None = None,
        pi: float | None = None,
        type: OptionType = "call",
        style: OptionStyle = "european",
        W: StateT | None = None,
        **kwargs: Any,
    ) -> None:
        expire = 1
        self.Rf = Rf
        self.k = k
        self.F = F
        _asset = forex_factory(steps=expire, X=X, F=F, Rf=Rf, u=u, d=d)
        _R = interest_factory(R=Rd, steps=expire)
        _pi = pi_factory(pi=pi, steps=expire, asset=_asset, R=_R)
        _derivative = OptionModel(
            expire=expire,
            pi=_pi,
            R=_R,
            states=W,
            strike=k * F,
            type=type,
            asset=_asset,
            style=style,
        )
        super().__init__(asset=_asset, R=_R, pi=_pi, derivative=_derivative)

    @property
    def H1(self) -> float:
        return 1 / self.Rf * super().H1
