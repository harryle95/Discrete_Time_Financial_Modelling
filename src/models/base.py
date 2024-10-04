from __future__ import annotations

from collections.abc import Mapping
from typing import Literal

from numpy.typing import ArrayLike

OptionType = Literal["put", "call"]
OptionStyle = Literal["european", "american"]
BarrierType = Literal["up and in", "up and out", "down and in", "down and out"]
ContractType = Literal["long", "short"]
NumberType = int | float
StateT = Mapping[int, ArrayLike]
