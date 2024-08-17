from src.helpers import calculate_H0, calculate_H1, calculate_W_0_general, calculate_W_0_replicating
from src.models import (
    OptionType,
    StatePriceModel,
    StateT,
    asset_factory,
    derivative_factory,
    interest_factory,
    pi_factory,
)

__all__ = ("Solver", "OneStepSolver")


class Solver:
    def __init__(
        self,
        expire: int,
        S: float | None = None,
        u: float | None = None,
        d: float | None = None,
        asset_states: StateT | None = None,
        derivative_states: StateT | None = None,
        strike: float | None = None,
        type: OptionType | None = None,
        interest_value: float | None = None,
        interest_states: StateT | None = None,
        pi_value: float | None = None,
        pi_states: StateT | None = None,
    ) -> None:
        self.expire = expire
        self.asset = asset_factory(steps=expire, S=S, u=u, d=d, states=asset_states)
        self.R = interest_factory(value=interest_value, states=interest_states, steps=expire)
        self.pi = pi_factory(value=pi_value, states=pi_states, steps=expire, asset=self.asset, R=self.R)
        self.derivative = derivative_factory(
            expire=expire, pi=self.pi, R=self.R, states=derivative_states, strike=strike, type=type, asset=self.asset
        )

    @property
    def state_price(self) -> StatePriceModel:
        self._state_price = StatePriceModel(pi=self.pi, R=self.R, steps=self.expire)
        return self._state_price

    @property
    def premium_state_price(self) -> float:
        return sum([i * j for i, j in zip(self.state_price[-1], self.derivative[-1], strict=True)])

    @property
    def premium(self) -> float:
        return self.derivative[0, 0]


class OneStepSolver(Solver):
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
