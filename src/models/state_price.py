from math import comb

from src.models.indexable import Indexable
from src.models.interest import InterestRateModel
from src.models.pi import PiModel

__all__ = ("StatePriceModel",)


class StatePriceModel(Indexable):
    def __init__(self, pi: PiModel, R: InterestRateModel, steps: int) -> None:
        self.pi = pi
        self.R = R
        super().__init__(steps)
        for n in range(self.steps + 1):
            self.set_state(
                n, [comb(n, j) * (pi[n, j] ** j) * ((1 - pi[n, j]) ** (n - j)) / R[n, j] ** n for j in range(n + 1)]
            )
