from math import comb
from typing import overload

from src.models.base import TerminalParams
from src.models.indexable import Indexable
from src.models.pi import Pi

__all__ = ("StatePrice", "TerminalStatePrice", "state_price_factory")


class StatePrice(Indexable):
    def __init__(self, steps: int) -> None:
        super().__init__(steps)

    def compute_grid(
        self,
        pi: Pi,
    ) -> None:
        for n in range(self.steps + 1):
            self.grid[n] = [
                comb(n, j) * (pi.p_up**j) * (pi.p_down ** (n - j)) / pi.R**n
                for j in range(n + 1)
            ]


class TerminalStatePrice(StatePrice):
    def __init__(self, steps: int, state: TerminalParams) -> None:
        self.state = state
        super().__init__(steps)
        self.set_terminal(state=state.V_T)


@overload
def state_price_factory(steps: int, state: TerminalParams) -> TerminalStatePrice: ...
@overload
def state_price_factory(steps: int, state: None = None) -> StatePrice: ...
def state_price_factory(
    steps: int, state: TerminalParams | None = None
) -> StatePrice | TerminalStatePrice:
    if state:
        return TerminalStatePrice(steps, state)
    return StatePrice(steps)
