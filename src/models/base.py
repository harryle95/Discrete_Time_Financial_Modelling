from collections.abc import Mapping
from enum import Enum
from typing import Literal

from numpy.typing import ArrayLike

StateT = Mapping[int, ArrayLike]


class _BarrierType(Enum):
    UP_IN = "up and in"
    UP_OUT = "up and out"
    DOWN_IN = "down and in"
    DOWN_OUT = "down and out"


OptionType = Literal["put", "call"]
OptionStyle = Literal["european", "american"]
BarrierType = _BarrierType | Literal["up and in", "up and out", "down and in", "down and out"]
ContractType = Literal["long", "short"]
