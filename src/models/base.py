from collections.abc import Mapping, Sequence
from enum import Enum
from typing import Literal

StateT = Mapping[int, Sequence[float]]


class _OptionStyle(Enum):
    EUROPEAN = "european"
    AMERICAN = "american"


class _OptionType(Enum):
    PUT = "put"
    CALL = "call"


OptionType = _OptionType | Literal["put", "call"]
OptionStyle = _OptionStyle | Literal["european", "american"]
