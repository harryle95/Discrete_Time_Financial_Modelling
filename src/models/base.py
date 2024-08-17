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
    """Terminal Known State"""


@dataclass
class PiParams:
    R: float
    """Interest Rate"""


@dataclass
class CRRParams:
    S_0: float
    """Base Asset Price in CRR model"""
    u: float
    """Up Factor"""
    d: float
    """Down Factor"""


@dataclass
class CRRPiParams(CRRParams, PiParams): ...


@dataclass
class BinomPiParams(PiParams):
    S_0: float
    """Original Asset Price"""
    S_11: float
    """Asset Price after one time step in the up state"""
    S_10: float
    """Asset Price after one time step in the down state"""


@dataclass
class TerminalPiParams(PiParams):
    p_up: float
    """Probability of an up sate in a binomial model"""
    p_down: float
    """Probability of a down sate in a binomial model"""


class Style(Enum):
    EUROPEAN = "european"
    AMERICAN = "american"


class OptionType(Enum):
    PUT = "put"
    CALL = "call"


@dataclass
class DerivativeParams:
    strike: float
    """Strike Price"""
    expire: int
    """Derivative Expire Period"""
    type: OptionType | Literal["put", "call"]
    """Option type - put/call"""
    style: Style | Literal["european", "american"] = "european"
    """Option style - European/American"""


@dataclass
class TerminalDerivativeParams(TerminalParams):
    expire: int
    """Derivative Expire Period"""


@dataclass
class AssetParams:
    steps: int
    """Number of steps in the asset model"""


@dataclass
class CRRAssetParams(AssetParams, CRRParams): ...


@dataclass
class TerminalAssetParams(AssetParams, TerminalParams):
    S_0: float
    """Asset price at time 0"""
