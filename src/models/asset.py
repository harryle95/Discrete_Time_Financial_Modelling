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

    def __init__(self, S: float, u: float, d: float, steps: int) -> None:
        super().__init__(steps=steps)
        self.S = S
        self.u = u
        self.d = d
        for t in range(self.steps + 1):
            self.set_state(t, [self.S * (self.u**index) * (self.d ** (t - index)) for index in range(t + 1)])


def asset_factory(
    steps: int,
    S: float | None = None,
    u: float | None = None,
    d: float | None = None,
    states: StateT | None = None,
) -> CRRAsset | StandardAsset:
    """Factory method to generate asset model

    Args:
        steps (int): number of steps
        S (float | None, optional): used for CRR model - initial price. Defaults to None.
        u (float | None, optional): used for CRR model - up factor. Defaults to None.
        d (float | None, optional): used for CRR model - down factor. Defaults to None.
        states (StateT | None, optional): used for state model - known state values. Defaults to None.

    Returns:
        CRRAsset | StandardAsset: model
    """
    if states:
        return StandardAsset(states=states, steps=steps)
    if not S or not u or not d:
        raise ValueError("CRR model expects non null S, u, d")
    return CRRAsset(S, u, d, steps)
