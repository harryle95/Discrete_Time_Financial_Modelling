from __future__ import annotations

from math import exp, log, sqrt
from statistics import NormalDist
from typing import Literal, cast

import scipy  # type: ignore[import-untyped]
import scipy.optimize  # type: ignore[import-untyped]

__all__ = (
    "calculate_BS_CRR_factor",
    "calculate_BS_Call",
    "calculate_BS_Put",
    "calculate_BS_R",
    "calculate_H0",
    "calculate_H1",
    "calculate_W_0_general",
    "calculate_W_0_replicating",
    "calculate_pi",
    "calculate_pi_CRR",
    "present_value",
    "put_call_parity",
)


StdNormal = NormalDist(0, 1)
NumberType = int | float


def present_value(future_value: NumberType, R: NumberType, T: NumberType) -> NumberType:
    """Calculate the present value based on future value, interest, and accumulating period

    Args:
        future_value (NumberType): future value
        R (NumberType): interest rate
        T (NumberType): accumulating period

    Returns:
        NumberType: present value
    """
    return future_value / R**T


def put_call_parity(asset: NumberType, strike: NumberType, R: NumberType, T: NumberType = 0) -> NumberType:
    """Calculate the rational put call parity. Will hold true at expire time for European derivatives.
    May not hold true at T = 0.

    The method automatically discount strike price to current value, based on R and T

    Args:
        asset (NumberType): asset at time T
        strike (NumberType): strike price at expiry.
        R (NumberType): interest rate
        T (NumberType, optional): accumulating period. When T = 0, asset is currently at expiry

    Returns:
        NumberType: _description_
    """
    return asset - present_value(strike, R, T)


def calculate_H1(
    S_11: NumberType,
    S_10: NumberType,
    W_11: NumberType,
    W_10: NumberType,
) -> NumberType:
    """Calculate H1 for one step binomial model"""
    return (W_11 - W_10) / (S_11 - S_10)


def calculate_H0(
    S_11: NumberType,
    S_10: NumberType,
    W_11: NumberType,
    W_10: NumberType,
    R: NumberType,
) -> NumberType:
    """Calculate H0 for one step binomial model"""
    return (S_11 * W_10 - S_10 * W_11) / (S_11 - S_10) / R


def calculate_pi(
    S_11: NumberType,
    S_10: NumberType,
    R: NumberType,
    S_0: NumberType,
) -> NumberType:
    """Calculate pi for one step binomial model"""
    return (R * S_0 - S_10) / (S_11 - S_10)


def calculate_pi_CRR(
    u: NumberType,
    d: NumberType,
    R: NumberType,
) -> NumberType:
    """Calculate pi for CRR model"""
    return (R - d) / (u - d)


def calculate_W_0_replicating(H0: NumberType, H1: NumberType, S_0: NumberType) -> NumberType:
    """Calculate option premium using the replicating portfolio method"""
    return H0 + H1 * S_0


def calculate_W_0_general(
    W_11: NumberType, W_10: NumberType, p_up: NumberType, p_down: NumberType, R: NumberType
) -> NumberType:
    """Calculate the option premium using the general pricing formula"""
    return 1 / R * (p_up * W_11 + p_down * W_10)


def calculate_BS_Call(S: NumberType, K: NumberType, r: NumberType, T: NumberType, sigma: NumberType) -> NumberType:
    """Calculate call value at time t = 0 using Black Scholes model

    Args:
        S (NumberType): asset price at t = 0
        K (NumberType): strike
        r (NumberType): continuous compounding interest rate - unit must be consistent with T
        T (NumberType): time NumberTypeerval T - unit must be consistent with r
        sigma (NumberType): volatility of the asset

    Returns:
        NumberType: call value
    """
    d1 = (log(S / K) + (r + sigma**2 / 2) * T) / (sigma * sqrt(T))
    d2 = d1 - sigma * sqrt(T)
    return S * StdNormal.cdf(d1) - K * exp(-r * T) * StdNormal.cdf(d2)


def calculate_BS_Put(S: NumberType, K: NumberType, r: NumberType, T: NumberType, sigma: NumberType) -> NumberType:
    """Calculate put value at time t = 0 using Black Scholes model

    Args:
        S (NumberType): asset price at t = 0
        K (NumberType): strike
        r (NumberType): continuous compounding interest rate - unit must be consistent with T
        T (NumberType): time NumberTypeerval T - unit must be consistent with r
        sigma (NumberType): volatility of the asset

    Returns:
        NumberType: put value
    """
    d1 = (log(S / K) + (r + sigma**2 / 2) * T) / (sigma * sqrt(T))
    d2 = d1 - sigma * sqrt(T)
    return -S * StdNormal.cdf(-d1) + K * exp(-r * T) * StdNormal.cdf(-d2)


def calculate_BS_CRR_factor(sigma: NumberType, T: NumberType, N: NumberType) -> tuple[NumberType, NumberType]:
    """Calculate CRR factor under BS model

    Args:
        sigma (NumberType): volatility
        T (NumberType): time period
        N (NumberType): number of time steps

    Returns:
        tuple[NumberType, NumberType]: _description_
    """
    u = exp(sigma * sqrt(T / N))
    d = 1 / u
    return u, d


def calculate_BS_R(r: NumberType, T: NumberType, N: NumberType) -> NumberType:
    """Calculate effective annual rate R based on compound interest rate r

    Args:
        r (NumberType): compound rate
        T (NumberType): period
        N (NumberType): time steps

    Returns:
        NumberType: rate
    """
    return exp(r * T / N)


def calculate_BS_r_from_R(R: NumberType, N: NumberType, T: NumberType) -> NumberType:
    return log(R**N) / T


def calculate_BS_sigma_from_u(u: NumberType, T: NumberType, N: NumberType) -> NumberType:
    return log(u) * sqrt(N / T)


def calculate_BS_sigma(
    S: NumberType,
    K: NumberType,
    T: NumberType,
    r: NumberType,
    W: NumberType,
    type: Literal["call", "put"],
    range: list[NumberType],
) -> NumberType:
    def wrapper(sigma: NumberType) -> NumberType:
        return calculate_BS_Call(S, K, r, T, sigma) - W if type == "call" else calculate_BS_Put(S, K, r, T, sigma) - W

    return cast(NumberType, scipy.optimize.root_scalar(wrapper, bracket=range, method="brentq").root)


def calculate_forward_rate(R: NumberType, S_0: NumberType) -> NumberType:
    return R * S_0


def calculate_forward_rate_forex(Rd: NumberType, Rf: NumberType, X_0: NumberType) -> NumberType:
    return Rd / Rf * X_0
