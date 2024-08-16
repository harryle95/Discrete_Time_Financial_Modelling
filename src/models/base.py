from collections.abc import Sequence
from dataclasses import dataclass
from enum import Enum
from typing import Literal

__all__ = (
    "PiParams",
    "CRRParams",
    "CRRPiParams",
    "Style",
    "OptionType",
    "DerivativeParams",
    "AssetParams",
    "CRRAssetParams",
    "TerminalParams",
    "TerminalAssetParams",
    "BinomPiParams",
    "TerminalDerivativeParams",
)


@dataclass
class TerminalParams:
    V_T: Sequence[float]


@dataclass
class PiParams:
    R: float


@dataclass
class CRRParams:
    S_0: float
    u: float
    d: float


@dataclass
class CRRPiParams(CRRParams, PiParams): ...


@dataclass
class BinomPiParams(PiParams):
    S_0: float
    S_11: float
    S_10: float


@dataclass
class TerminalPiParams(PiParams):
    p_up: float
    p_down: float


class Style(Enum):
    EUROPEAN = "european"
    AMERICAN = "american"


class OptionType(Enum):
    PUT = "put"
    CALL = "call"


@dataclass
class DerivativeParams:
    strike: float
    expire: int = 1
    style: Style | Literal["european", "american"] = "european"
    type: OptionType | Literal["put", "call"] = "call"


@dataclass
class TerminalDerivativeParams(TerminalParams):
    expire: int = 1


@dataclass
class AssetParams:
    steps: int


@dataclass
class CRRAssetParams(AssetParams, CRRParams): ...


@dataclass
class TerminalAssetParams(AssetParams, TerminalParams):
    S_0: float
