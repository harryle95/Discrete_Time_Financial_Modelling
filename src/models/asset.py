from typing import overload

from src.models.base import CRRAssetParams, TerminalAssetParams
from src.models.indexable import Indexable

__all__ = ("Asset", "CRRAsset", "TerminalAsset", "asset_factory")


class Asset(Indexable):
    def __init__(self, S_0: float, steps: int) -> None:
        self.S_0 = S_0
        self.steps = steps
        super().__init__(steps=self.steps)
        self.grid[0] = [S_0]


class CRRAsset(Asset):
    def __init__(self, params: CRRAssetParams) -> None:
        self.u = params.u
        self.d = params.d
        super().__init__(params.S_0, params.steps)
        self.compute_grid()

    def compute_grid(self) -> None:
        for t in range(self.steps + 1):
            self.grid[t] = [
                self.S_0 * (self.u**index) * (self.d ** (t - index))
                for index in range(t + 1)
            ]


class TerminalAsset(Asset):
    def __init__(self, params: TerminalAssetParams) -> None:
        super().__init__(params.S_0, params.steps)
        self.set_terminal(params.V_T)


@overload
def asset_factory(params: CRRAssetParams) -> CRRAsset: ...
@overload
def asset_factory(params: TerminalAssetParams) -> TerminalAsset: ...
def asset_factory(
    params: CRRAssetParams | TerminalAssetParams,
) -> CRRAsset | TerminalAsset:
    if isinstance(params, CRRAssetParams):
        return CRRAsset(params)
    if isinstance(params, TerminalAssetParams):
        return TerminalAsset(params)
