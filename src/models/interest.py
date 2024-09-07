from src.models.base import StateT
from src.models.indexable import Constant, Indexable

__all__ = (
    "ConstantInterestRate",
    "InterestRateModel",
    "VariableInterestRate",
    "interest_factory",
)


class InterestRateModel(Indexable): ...


class ConstantInterestRate(Constant, InterestRateModel):
    def __init__(self, R: float | int) -> None:
        super().__init__(value=R)


class VariableInterestRate(InterestRateModel):
    def __init__(self, R: StateT, steps: int) -> None:
        super().__init__(steps)
        if 0 not in R:
            raise ValueError("Variable Interest Rate model requires current asset value at t=0")
        for time, state in R.items():
            self.set_state(time, state)


def interest_factory(
    R: float | int | StateT,
    steps: int | None = None,
) -> ConstantInterestRate | VariableInterestRate:
    if isinstance(R, float | int):
        return ConstantInterestRate(R)
    if not steps:
        raise ValueError("Variable interest rate requires knowing steps")
    return VariableInterestRate(R=R, steps=steps)
