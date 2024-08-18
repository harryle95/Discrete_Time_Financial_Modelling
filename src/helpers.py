from math import exp, log, sqrt
from statistics import NormalDist
from typing import Literal, cast

import scipy  # type: ignore
import scipy.optimize  # type: ignore

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


def present_value(future_value: float, R: float, T: float) -> float:
    """Calculate the present value based on future value, interest, and accumulating period

    Args:
        future_value (float): future value
        R (float): interest rate
        T (float): accumulating period

    Returns:
        float: present value
    """
    return float(future_value / R**T)


def put_call_parity(asset: float, strike: float, R: float, T: float = 0) -> float:
    """Calculate the rational put call parity. Will hold true at expire time for European derivatives.
    May not hold true at T = 0.

    The method automatically discount strike price to current value, based on R and T

    Args:
        asset (float): asset at time T
        strike (float): strike price at expiry.
        R (float): interest rate
        T (float, optional): accumulating period. When T = 0, asset is currently at expiry

    Returns:
        float: _description_
    """
    return asset - present_value(strike, R, T)


def calculate_H1(
    S_11: float,
    S_10: float,
    W_11: float,
    W_10: float,
) -> float:
    """Calculate H1 for one step binomial model"""
    return (W_11 - W_10) / (S_11 - S_10)


def calculate_H0(
    S_11: float,
    S_10: float,
    W_11: float,
    W_10: float,
    R: float,
) -> float:
    """Calculate H0 for one step binomial model"""
    return (S_11 * W_10 - S_10 * W_11) / (S_11 - S_10) / R


def calculate_pi(
    S_11: float,
    S_10: float,
    R: float,
    S_0: float,
) -> float:
    """Calculate pi for one step binomial model"""
    return (R * S_0 - S_10) / (S_11 - S_10)


def calculate_pi_CRR(
    u: float,
    d: float,
    R: float,
) -> float:
    """Calculate pi for CRR model"""
    return (R - d) / (u - d)


def calculate_W_0_replicating(H0: float, H1: float, S_0: float) -> float:
    """Calculate option premium using the replicating portfolio method"""
    return H0 + H1 * S_0


def calculate_W_0_general(W_11: float, W_10: float, p_up: float, p_down: float, R: float) -> float:
    """Calculate the option premium using the general pricing formula"""
    return 1 / R * (p_up * W_11 + p_down * W_10)


def calculate_BS_Call(S: float, K: float, r: float, T: float, sigma: float) -> float:
    """Calculate call value at time t = 0 using Black Scholes model

    Args:
        S (float): asset price at t = 0
        K (float): strike
        r (float): continuous compounding interest rate - unit must be consistent with T
        T (float): time interval T - unit must be consistent with r
        sigma (float): volatility of the asset

    Returns:
        float: call value
    """
    d1 = (log(S / K) + (r + sigma**2 / 2) * T) / (sigma * sqrt(T))
    d2 = d1 - sigma * sqrt(T)
    return S * StdNormal.cdf(d1) - K * exp(-r * T) * StdNormal.cdf(d2)


def calculate_BS_Put(S: float, K: float, r: float, T: float, sigma: float) -> float:
    """Calculate put value at time t = 0 using Black Scholes model

    Args:
        S (float): asset price at t = 0
        K (float): strike
        r (float): continuous compounding interest rate - unit must be consistent with T
        T (float): time interval T - unit must be consistent with r
        sigma (float): volatility of the asset

    Returns:
        float: put value
    """
    d1 = (log(S / K) + (r + sigma**2 / 2) * T) / (sigma * sqrt(T))
    d2 = d1 - sigma * sqrt(T)
    return -S * StdNormal.cdf(-d1) + K * exp(-r * T) * StdNormal.cdf(-d2)


def calculate_BS_CRR_factor(sigma: float, T: float, N: float) -> tuple[float, float]:
    """Calculate CRR factor under BS model

    Args:
        sigma (float): volatility
        T (float): time period
        N (float): number of time steps

    Returns:
        tuple[float, float]: _description_
    """
    u = exp(sigma * sqrt(T / N))
    d = 1 / u
    return u, d


def calculate_BS_R(r: float, T: float, N: float) -> float:
    """Calculate effective annual rate R based on compound interest rate r

    Args:
        r (float): compound rate
        T (float): period
        N (float): time steps

    Returns:
        float: rate
    """
    return exp(r * T / N)


def calculate_BS_r_from_R(R: float, N: int, T: float) -> float:
    return log(R**N) / T


def calculate_BS_sigma_from_u(u: float, T: float, N: int) -> float:
    return log(u) * sqrt(N / T)


def calculate_BS_sigma(
    S: float, K: float, T: float, r: float, W: float, type: Literal["call", "put"], range: list[float]
) -> float:
    def wrapper(sigma: float) -> float:
        return calculate_BS_Call(S, K, r, T, sigma) - W if type == "call" else calculate_BS_Put(S, K, r, T, sigma) - W

    return cast(float, scipy.optimize.root_scalar(wrapper, bracket=range, method="brentq").root)
