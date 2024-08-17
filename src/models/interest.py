from src.models.base import StateT
from src.models.indexable import Constant, Indexable


class InterestRateModel(Indexable): ...


class ConstantInterestRate(Constant, InterestRateModel):
    def __init__(self, value: float) -> None:
        self.value = value


class VariableInterestRate(InterestRateModel):
    def __init__(self, states: StateT, steps: int) -> None:
        super().__init__(steps)
        if 0 not in states:
            raise ValueError("Asset Model requires current asset value at t=0")
        for time, state in states.items():
            self.set_state(time, state)


def interest_factory(
    value: float | None = None,
    states: StateT | None = None,
    steps: int | None = None,
) -> ConstantInterestRate | VariableInterestRate:
    if value:
        return ConstantInterestRate(value)
    if not states or not steps:
        raise ValueError("Variable interest rate requires knowing states and steps")
    return VariableInterestRate(states=states, steps=steps)
