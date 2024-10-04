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
from src.models.base import BarrierType, NumberType
from src.models.derivative import BarrierOption, Option

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
    def premium_state_price(self) -> NumberType:
        return cast(NumberType, sum([i * j for i, j in zip(self.state_price[-1], self.derivative[-1], strict=True)]))


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
    def H0(self) -> NumberType:
        return calculate_H0(
            self.asset[1, 1],
            self.asset[1, 0],
            self.derivative[1, 1],
            self.derivative[1, 0],
            self.R[0, 0],
        )

    @property
    def H1(self) -> NumberType:
        return calculate_H1(
            self.asset[1, 1],
            self.asset[1, 0],
            self.derivative[1, 1],
            self.derivative[1, 0],
        )

    @property
    def premium(self) -> NumberType:
        return calculate_W_0_general(
            self.derivative[1, 1],
            self.derivative[1, 0],
            self.pi[0, 0],
            1 - self.pi[0, 0],
            self.R[0, 0],
        )

    @property
    def premium_replicating(self) -> NumberType:
        return calculate_W_0_replicating(self.H0, self.H1, self.asset[0, 0])


class AssetOptionSolver(OptionSolver):
    @overload
    def __init__(
        self,
        *,
        expire: int,
        S: NumberType,
        u: NumberType,
        d: NumberType,
        W: StateT | None = None,
        K: NumberType = 0,
        type: OptionType = "call",
        style: OptionStyle = "european",
        R: NumberType | StateT = 1.0,
        **kwargs: Any,
    ) -> None: ...
    @overload
    def __init__(
        self,
        *,
        expire: int,
        S: StateT,
        pi: NumberType | StateT,
        W: StateT | None = None,
        K: NumberType = 0,
        type: OptionType = "call",
        style: OptionStyle = "european",
        R: NumberType | StateT = 1.0,
        **kwargs: Any,
    ) -> None: ...

    def __init__(
        self,
        *,
        expire: int,
        S: NumberType | StateT,
        u: NumberType | None = None,
        d: NumberType | None = None,
        W: StateT | None = None,
        K: NumberType = 0,
        type: OptionType = "call",
        style: OptionStyle = "european",
        R: NumberType | StateT = 1.0,
        pi: NumberType | StateT | None = None,
        **kwargs: Any,
    ) -> None:
        _asset = asset_factory(steps=expire, S=S, u=u, d=d)
        _R = interest_factory(R=R, steps=expire)
        _pi = pi_factory(pi=pi, steps=expire, asset=_asset, R=_R)
        _derivative = Option(
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


class BarrierAssetOptionSolver(OptionSolver):
    @overload
    def __init__(
        self,
        *,
        expire: int,
        S: NumberType,
        u: NumberType,
        d: NumberType,
        K: NumberType = 0,
        type: OptionType = "call",
        style: OptionStyle = "european",
        R: NumberType | StateT = 1.0,
        B: NumberType,
        barrier_type: BarrierType = "up and in",
        **kwargs: Any,
    ) -> None: ...
    @overload
    def __init__(
        self,
        *,
        expire: int,
        S: StateT,
        pi: NumberType | StateT,
        K: NumberType = 0,
        type: OptionType = "call",
        style: OptionStyle = "european",
        R: NumberType | StateT = 1.0,
        B: NumberType,
        barrier_type: BarrierType = "up and in",
        **kwargs: Any,
    ) -> None: ...

    def __init__(
        self,
        *,
        expire: int,
        B: NumberType,
        barrier_type: BarrierType = "up and in",
        S: NumberType | StateT,
        u: NumberType | None = None,
        d: NumberType | None = None,
        K: NumberType = 0,
        type: OptionType = "call",
        style: OptionStyle = "european",
        R: NumberType | StateT = 1.0,
        pi: NumberType | StateT | None = None,
        **kwargs: Any,
    ) -> None:
        _asset = asset_factory(steps=expire, S=S, u=u, d=d)
        _R = interest_factory(R=R, steps=expire)
        _pi = pi_factory(pi=pi, steps=expire, asset=_asset, R=_R)
        _derivative = BarrierOption(
            expire=expire,
            barrier=B,
            pi=_pi,
            R=_R,
            strike=K,
            type=type,
            asset=_asset,
            style=style,
            barrier_type=barrier_type,
        )
        super().__init__(expire=expire, asset=_asset, R=_R, pi=_pi, derivative=_derivative)


class OneStepAssetOptionSolver(OneStepOptionSolver):
    @overload
    def __init__(
        self,
        *,
        S: NumberType,
        u: NumberType,
        d: NumberType,
        K: NumberType = 0,
        type: OptionType = "call",
        style: OptionStyle = "european",
        R: NumberType | StateT = 1.0,
        pi: NumberType | None = None,
        **kwargs: Any,
    ) -> None: ...
    @overload
    def __init__(
        self,
        *,
        S: StateT,
        W: StateT | None = None,
        R: NumberType | StateT = 1.0,
        pi: NumberType | None = None,
        **kwargs: Any,
    ) -> None: ...

    def __init__(
        self,
        *,
        S: NumberType | StateT | NumberType,
        u: NumberType | None = None,
        d: NumberType | None = None,
        W: StateT | None = None,
        K: NumberType = 0,
        type: OptionType = "call",
        style: OptionStyle = "european",
        R: NumberType | StateT = 1.0,
        pi: NumberType | None = None,
        **kwargs: Any,
    ) -> None:
        _asset = asset_factory(steps=1, S=S, u=u, d=d)
        _R = interest_factory(R=R, steps=1)
        _pi = pi_factory(pi=pi, steps=1, asset=_asset, R=_R)
        _derivative = Option(
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


class ForexOptionSolver(OptionSolver):
    def __init__(
        self,
        *,
        expire: int,
        X: NumberType | StateT,
        F: NumberType,
        Rf: NumberType,
        Rd: NumberType,
        k: NumberType,
        u: NumberType | None = None,
        d: NumberType | None = None,
        pi: NumberType | None = None,
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
        _derivative = Option(
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


class OneStepForexOptionSolver(OneStepOptionSolver):
    def __init__(
        self,
        *,
        X: NumberType | StateT,
        F: NumberType,
        Rf: NumberType,
        Rd: NumberType,
        k: NumberType,
        u: NumberType | None = None,
        d: NumberType | None = None,
        pi: NumberType | None = None,
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
        _derivative = Option(
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
    def H1(self) -> NumberType:
        return 1 / self.Rf * super().H1
