from collections.abc import Mapping, Sequence
from enum import Enum
from typing import Literal

StateT = Mapping[int, Sequence[float | int]]


class _OptionStyle(Enum):
    EUROPEAN = "european"
    AMERICAN = "american"


class _OptionType(Enum):
    PUT = "put"
    CALL = "call"


class _BarrierType(Enum):
    UP_IN = "up and in"
    UP_OUT = "up and out"
    DOWN_IN = "down and in"
    DOWN_OUT = "down and out"


OptionType = _OptionType | Literal["put", "call"]
OptionStyle = _OptionStyle | Literal["european", "american"]
BarrierType = _BarrierType | Literal["up and in", "up and out", "down and in", "down and out"]
