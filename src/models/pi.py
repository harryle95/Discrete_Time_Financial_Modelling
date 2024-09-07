from src.helpers import calculate_pi, calculate_pi_CRR
from src.models.asset import AssetModel, CRRAsset
from src.models.base import StateT
from src.models.indexable import Constant, Indexable
from src.models.interest import ConstantInterestRate, InterestRateModel

__all__ = (
    "ConstantPi",
    "PiModel",
    "StatePi",
    "VariablePi",
    "pi_factory",
)


class PiModel(Indexable): ...


class ConstantPi(Constant, PiModel):
    def __init__(self, value: float) -> None:
        super().__init__(value=value)


class VariablePi(PiModel):
    def __init__(self, R: InterestRateModel, asset: AssetModel):
        super().__init__(steps=asset.steps - 1)
        for n in range(self.steps + 1):
            self.set_state(
                n,
                [
                    (R[n, j] * asset[n, j] - asset[n + 1, j]) / (asset[n + 1, j + 1] - asset[n + 1, j])
                    for j in range(n + 1)
                ],
            )


class StatePi(PiModel):
    def __init__(self, pi: StateT, steps: int) -> None:
        super().__init__(steps)
        if 0 not in pi:
            raise ValueError("Asset Model requires current asset value at t=0")
        for time, state in pi.items():
            self.set_state(time, state)


def pi_factory(
    pi: float | StateT | None = None,
    steps: int | None = None,
    asset: AssetModel | None = None,
    R: InterestRateModel | None = None,
) -> ConstantPi | VariablePi | StatePi:
    if isinstance(pi, float):
        return ConstantPi(pi)
    if isinstance(pi, dict) and steps:
        return StatePi(pi=pi, steps=steps)
    if isinstance(asset, CRRAsset) and isinstance(R, ConstantInterestRate):
        pi = calculate_pi_CRR(asset.u, asset.d, R.value)
        return ConstantPi(pi)
    if asset and isinstance(R, ConstantInterestRate):
        pi = calculate_pi(asset[1, 1], asset[1, 0], R.value, asset[0, 0])
        return ConstantPi(pi)
    if asset and R:
        return VariablePi(R, asset)
    raise ValueError("Expect at least value or states and steps or asset and R to be provided")
