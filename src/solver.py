from __future__ import annotations

from typing import Any, cast, overload

from src.helpers import calculate_H0, calculate_H1, calculate_W_0_general, calculate_W_0_replicating
from src.models import (
    DerivativeModel,
    OptionStyle,
    OptionType,
    StatePriceModel,
    StateT,
    asset_factory,
    interest_factory,
    pi_factory,
)

__all__ = ("Solver", "OneStepSolver")


class Solver:
    @overload
    @classmethod
    def init(
        cls,
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
    ) -> Solver: ...
    @overload
    @classmethod
    def init(
        cls,
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
    ) -> Solver: ...
    @classmethod
    def init(
        cls,
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
    ) -> Solver:
        return cls(expire=expire, S=S, u=u, d=d, W=W, K=K, type=type, style=style, R=R, pi=pi, **kwargs)

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
        self.expire = expire
        self.asset = asset_factory(steps=expire, S=S, u=u, d=d)
        self.R = interest_factory(R=R, steps=expire)
        self.pi = pi_factory(pi=pi, steps=expire, asset=self.asset, R=self.R)
        self.derivative = DerivativeModel(
            expire=expire,
            pi=self.pi,
            R=self.R,
            states=W,
            strike=K,
            type=type,
            asset=self.asset,
            style=style,
        )

    @property
    def state_price(self) -> StatePriceModel:
        self._state_price = StatePriceModel(pi=self.pi, R=self.R, steps=self.expire)
        return self._state_price

    @property
    def premium_state_price(self) -> float:
        return cast(float, sum([i * j for i, j in zip(self.state_price[-1], self.derivative[-1], strict=True)]))


class OneStepSolver(Solver):
    @overload
    @classmethod
    def init(
        cls,
        *,
        S: float | int,
        u: float,
        d: float,
        K: float = 0,
        type: OptionType = "call",
        style: OptionStyle = "european",
        R: float | StateT = 1.0,
        **kwargs: Any,
    ) -> OneStepSolver: ...
    @overload
    @classmethod
    def init(
        cls,
        *,
        S: StateT,
        W: StateT | None = None,
        R: float | StateT = 1.0,
        **kwargs: Any,
    ) -> OneStepSolver: ...
    @classmethod
    def init(
        cls,
        *,
        S: float | StateT | int,
        u: float | None = None,
        d: float | None = None,
        W: StateT | None = None,
        K: float = 0,
        type: OptionType = "call",
        style: OptionStyle = "european",
        R: float | StateT = 1.0,
        pi: float | StateT | None = None,
        **kwargs: Any,
    ) -> OneStepSolver:
        return cls(expire=1, S=S, u=u, d=d, W=W, K=K, type=type, style=style, R=R, pi=pi, **kwargs)

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
