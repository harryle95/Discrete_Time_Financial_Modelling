import math
from collections.abc import Sequence
from typing import cast

import numpy as np
import pytest
from numpy.testing import assert_almost_equal, assert_approx_equal

from src.helpers import (
    calculate_BS_Call,
    calculate_BS_CRR_factor,
    calculate_BS_Put,
    calculate_BS_R,
    calculate_BS_r_from_R,
    calculate_BS_sigma,
    calculate_BS_sigma_from_u,
    calculate_forward_rate,
    calculate_forward_rate_forex,
    present_value,
    put_call_parity,
)
from src.models import (
    ConstantPi,
    OptionType,
    asset_factory,
    interest_factory,
    pi_factory,
)
from src.models.base import BarrierType, NumberType, OptionStyle, OptionType
from src.models.derivative import BarrierOption, Option
from src.solver import AssetOptionSolver, BarrierAssetOptionSolver, ForexOptionSolver, OneStepAssetOptionSolver


@pytest.mark.parametrize(
    "K, asset, type, exp_value",
    [
        (22, 35, "call", 13),  # Quiz 1-7
        (14, 5, "put", 9),  # Quiz 1-8
    ],
)
def test_quiz_1_7(
    K: NumberType,
    asset: NumberType,
    type: OptionType,
    exp_value: NumberType,
) -> None:
    value = Option.exercised_value(K=K, S=asset, type=type)
    assert_approx_equal(value, exp_value)


def test_quiz_1_9() -> None:
    N = 100
    P_0 = 1.47
    K = 14
    R = 1.03
    S_1 = 7
    value = Option.exercised_value(K=K, S=S_1, type="put")
    assert_approx_equal((value - P_0 * R) * N, 548.59)


def test_quiz_1_10() -> None:
    N = 100
    K = 42
    C_0 = 0.55
    R = 1.03
    S_1 = 36.0
    value = Option.exercised_value(K=K, S=S_1, type="call")
    profit = C_0 * N * R - N * value
    assert_approx_equal(profit, 56.65)


def test_example_1_13() -> None:
    R = 10 / 9
    S_11 = 20 / 3
    W_11 = 7.0
    S_10 = 40 / 9
    W_10 = 2.0
    S_0 = 5.0
    model = OneStepAssetOptionSolver(S={0: [S_0], 1: [S_10, S_11]}, W={1: [W_10, W_11]}, R=R)
    H0 = model.H0
    H1 = model.H1
    W0 = model.premium_replicating
    assert_approx_equal(-7.2, H0)
    assert_approx_equal(2.25, H1)
    assert_approx_equal(4.05, W0)


def test_example_1_14() -> None:
    K = 4
    R = 6 / 5
    S_11 = 6
    S_10 = 3
    S_0 = 4
    model = OneStepAssetOptionSolver(K=K, R=R, S={0: [S_0], 1: [S_10, S_11]}, type="call")
    H0 = model.H0
    H1 = model.H1
    W0 = model.premium_replicating
    assert_approx_equal(-5 / 3, H0)
    assert_approx_equal(2 / 3, H1)
    assert_approx_equal(1, W0)


def test_example_1_18() -> None:
    R = 10 / 9
    S_11 = 20 / 3
    W_11 = 7
    S_10 = 40 / 9
    W_10 = 2
    S_0 = 5
    model = OneStepAssetOptionSolver(R=R, S={0: [S_0], 1: [S_10, S_11]}, W={1: [W_10, W_11]})

    p_up = model.pi[0, 0]
    p_down = 1 - p_up
    W0 = model.derivative.premium
    assert_approx_equal(p_up, 1 / 2)
    assert_approx_equal(p_down, 1 / 2)
    assert_approx_equal(W0, 4.05)


def test_workshop_2_1() -> None:
    S_0 = 4.0
    S_11 = 8.0
    S_10 = 2.0
    r = 0.25
    K = 6.0
    R = 1 + r

    # Call Model
    call_model = OneStepAssetOptionSolver(
        S={0: [S_0], 1: [S_10, S_11]},
        R=R,
        K=K,
        type="call",
    )
    C_0_replicate = call_model.premium_replicating
    C_0_general = call_model.derivative.premium
    H0 = call_model.H0
    H1 = call_model.H1
    call_p_up = call_model.pi[0, 0]
    call_p_down = 1 - call_model.pi[0, 0]

    assert_approx_equal(-8 / 15, H0)
    assert_approx_equal(1 / 3, H1)
    assert_approx_equal(C_0_replicate, 0.8)

    assert_approx_equal(call_p_up, 1 / 2)
    assert_approx_equal(call_p_down, 1 / 2)
    assert_approx_equal(C_0_general, 0.8)

    # Put Model
    put_model = OneStepAssetOptionSolver(
        S={0: [S_0], 1: [S_10, S_11]},
        R=R,
        K=6,
        type="put",
    )
    P_0 = put_model.derivative.premium
    assert_approx_equal(P_0, 1.6)
    # Parity
    actual = C_0_general - P_0
    expected = put_call_parity(S_0, K, R, 1)
    assert_approx_equal(actual, -0.8)
    assert_approx_equal(expected, -0.8)


@pytest.mark.parametrize("W_11, S_11, W_10, S_10, R, T, H0, H1", [(0, 50, 5, 10, 1.25, 1, 5, -0.125)])
def test_quiz_2_1(
    W_11: NumberType,
    S_11: NumberType,
    W_10: NumberType,
    S_10: NumberType,
    R: NumberType,
    T: int,
    H0: NumberType,
    H1: NumberType,
) -> None:
    model = OneStepAssetOptionSolver(S={0: [0], 1: [S_10, S_11]}, W={1: [W_10, W_11]}, R=R)
    assert_approx_equal(H0, model.H0)
    assert_approx_equal(H1, model.H1)


def test_quiz_2_2() -> None:
    R = 1.1
    future = ((109, 1), (134, 3), (100, 0), (122, 2))
    values = {fv: present_value(fv[0], R, fv[1]) for fv in future}
    s_values = sorted(values.items(), key=lambda item: item[1], reverse=True)
    assert_almost_equal(s_values[0][0], (122, 2))


@pytest.mark.parametrize(
    "S_0, u, d, R, T, pi",
    [
        (18.0, 1.18, 0.98, 1.08, 1, 0.5),
        (12.0, 1.11, 0.94, 1.08, 1, 0.8235),
    ],
)
def test_quiz_2_3(S_0: NumberType, u: NumberType, d: NumberType, R: NumberType, T: int, pi: NumberType) -> None:
    asset = asset_factory(T, S_0, u, d)
    interest = interest_factory(R=R)
    pi_model = pi_factory(asset=asset, R=interest)
    assert_approx_equal(pi_model[0, 0], pi, 4)


def test_quiz_2_4() -> None:
    S_0 = 20.0
    u = 1.1
    d = 0.8
    model = asset_factory(1, S_0, u, d)
    assert_approx_equal(model[1, 1], 22)
    assert_approx_equal(model[1, 0], 16)


@pytest.mark.parametrize(
    "K, S_11, S_10, R, pi, T, type, premium",
    [
        (15, 24, 11, 1.05, 0.64, 1, "put", 1.371),
        (18, 25, 12, 1.05, 0.68, 1, "put", 1.829),
        (21, 34, 15, 1.02, 0.48, 1, "call", 6.118),
        (20, 38, 19, 1.05, 0.48, 1, "call", 8.229),
    ],
)
def test_quiz_2_5_6(
    K: NumberType,
    S_11: NumberType,
    S_10: NumberType,
    R: NumberType,
    pi: NumberType,
    T: int,
    type: OptionType,
    premium: NumberType,
) -> None:
    model = OneStepAssetOptionSolver(K=K, S={0: [0], 1: [S_10, S_11]}, R=R, pi=pi, type=type)
    assert_approx_equal(model.derivative.premium, premium, 4)


@pytest.mark.parametrize("K, C_0, S_0, R, premium", [(32, 2, 26, 1.02, 7.373), (34, 2.65, 27, 1.08, 7.131)])
def test_quiz_2_7(K: NumberType, C_0: NumberType, S_0: NumberType, R: NumberType, premium: NumberType) -> None:
    parity = put_call_parity(S_0, K, R, 1)
    P_0 = C_0 - parity
    assert_approx_equal(P_0, premium, 4)


@pytest.mark.parametrize(
    "S_0, u, R, profit",
    [
        (17.0, 1.07, 1.02, 1.452),
        (18.0, 1.08, 1.03, 1.873),
    ],
)
def test_quiz_2_8(S_0: NumberType, u: NumberType, R: NumberType, profit: NumberType) -> None:
    d = 1 / u
    model = asset_factory(1, S_0, u, d)
    S_10 = model[1, 0]
    actual_profit = S_0 * R - S_10
    assert_approx_equal(actual_profit, profit, 4)


@pytest.mark.parametrize(
    "W_0, S_0, W_11, W_10, S_11, S_10, N, R, H1",
    [
        (1.35, 44, 0.45, 4.72, 56, 24, 100, 1, -13.3),
        (1.08, 37, 0.48, 4.58, 59, 28, 100, 1, -13.2),
    ],
)
def test_quiz_2_10(
    W_0: NumberType,
    S_0: NumberType,
    W_11: NumberType,
    W_10: NumberType,
    S_11: NumberType,
    S_10: NumberType,
    N: NumberType,
    R: NumberType,
    H1: NumberType,
) -> None:
    model = OneStepAssetOptionSolver(
        S={0: [S_0], 1: [S_10, S_11]},
        W={0: [W_0], 1: [W_10 * N, W_11 * N]},
        R=R,
    )

    assert_approx_equal(model.H1, H1, 3)


@pytest.mark.parametrize(
    "S, u, d, T, result",
    [(4, 2, 1 / 2, 3, [[4.0], [2.0, 8.0], [1.0, 4.0, 16.0], [1 / 2, 2, 8, 32]])],
)
def test_example_2_1(S: NumberType, u: NumberType, d: NumberType, T: int, result: Sequence[NumberType]) -> None:
    model = asset_factory(T, S, u, d)
    for i in range(T + 1):
        assert_almost_equal(model[i], result[i])


def test_example_2_2() -> None:
    K = 3
    u = 2
    d = 1 / 2
    S = 4
    R = 5 / 4
    expire = 2
    model = AssetOptionSolver(expire=expire, K=K, u=u, d=d, S=S, R=R, type="call")
    assert_almost_equal(model.derivative[0], [2.40])
    assert_almost_equal(model.derivative[1], [0.4, 5.6])
    assert_almost_equal(model.derivative[2], [0, 1, 13])


def test_example_2_3() -> None:
    K = 3
    u = 2
    d = 1 / 2
    S = 4
    R = 5 / 4
    expire = 3
    model = AssetOptionSolver(expire=expire, K=K, u=u, d=d, S=S, R=R, type="call")
    derivative = model.derivative
    assert_almost_equal(derivative[0], [2.816])
    assert_almost_equal(derivative[1], [0.8, 6.24])
    assert_almost_equal(derivative[2], [0, 2, 13.60])
    assert_almost_equal(derivative[3], [0, 0, 5, 29])


def test_example_2_4() -> None:
    K = 3
    u = 2
    d = 1 / 2
    S = 4
    R = 5 / 4
    expire = 3
    model = AssetOptionSolver(expire=expire, K=K, u=u, d=d, S=S, R=R, type="call")
    derivative = model.derivative
    assert_almost_equal(model.state_price[3], [8 / 125, 24 / 125, 24 / 125, 8 / 125])
    assert_almost_equal(model.derivative.premium, derivative[0, 0])


def test_workshop_3_1() -> None:
    K = 23
    S = 20
    u = 1.06
    d = 0.95
    R = 1.01
    T = 10
    put_model = AssetOptionSolver(expire=T, K=K, S=S, u=u, d=d, R=R, type="put")
    put_premium = put_model.derivative.premium

    call_model = AssetOptionSolver(expire=T, K=K, S=S, u=u, d=d, R=R, type="call")
    call_premium = call_model.derivative.premium
    assert_almost_equal(put_premium, 1.8202, 4)
    assert_almost_equal(call_premium, 0.9986, 4)
    assert_approx_equal(put_call_parity(S, K, R, T), call_premium - put_premium)


def test_workshop_3_2() -> None:
    K = 10
    S = 10
    u = 1.2
    d = 1 / u
    T = 8
    R = 1.05
    put_model = AssetOptionSolver(expire=T, K=K, S=S, u=u, d=d, R=R, type="put")
    assert_almost_equal(
        put_model.state_price[-1],
        [
            0.000531,
            0.006135,
            0.031017,
            0.089606,
            0.161788,
            0.186955,
            0.135023,
            0.055724,
            0.010061,
        ],
        6,
    )
    assert_almost_equal(put_model.derivative.premium, 0.4793, 4)
    assert_almost_equal(put_model.derivative[0, 0], 0.4793, 4)


@pytest.mark.parametrize("S, u, d, T, largest, smallest", [(12, 1.2, 0.9, 3, 20.736, 8.748)])
def test_quiz_3_3(S: NumberType, u: NumberType, d: NumberType, T: int, largest: NumberType, smallest: NumberType) -> None:
    asset = asset_factory(T, S, u, d)
    assert_almost_equal(asset[T, T], largest)
    assert_almost_equal(asset[T, 0], smallest)


@pytest.mark.parametrize(
    "C, pi, R, T, premium",
    [([0, 0, 0, 3.84], 0.3, 1.13, 3, 0.0719), ([0, 0, 0, 3.5], 0.3, 1.18, 3, 0.0575)],
)
def test_quiz_3_4(C: Sequence[NumberType], pi: NumberType, R: NumberType, T: int, premium: NumberType) -> None:
    model = AssetOptionSolver(expire=T, W={T: C}, pi=pi, R=R, S={0: [0]})
    assert_almost_equal(model.derivative.premium, premium, 4)


@pytest.mark.parametrize("K, T, asset, put", [(30, 3, [10, 15, 25, 35], [20, 15, 5, 0])])
def test_quiz_3_5(K: NumberType, T: int, asset: Sequence[NumberType], put: NumberType) -> None:
    asset_model = asset_factory(S={0: [0], T: asset}, steps=T)
    pi = pi_factory(pi=0.1)
    R = interest_factory(R=0.1)
    derivative = Option(expire=T, strike=K, type="put", pi=pi, R=R, asset=asset_model)
    assert_almost_equal(derivative.final, put)


@pytest.mark.parametrize("R, state_22, result", [(1.2, 0.27, 0.0984), (1.3, 0.12, 0.1788)])
def test_quiz_3_6(R: NumberType, state_22: NumberType, result: NumberType) -> None:
    pi = math.sqrt(state_22 * R**2)
    pi_down = 1 - pi
    state_20 = (pi_down) ** 2 / R**2
    assert_almost_equal(state_20, result, 4)


@pytest.mark.parametrize(
    "derivative, R, pi, premium",
    [([0, 0, 1, 0], 1.06, 0.5, 0.3149), ([0, 0, 1, 0], 1.05, 0.7, 0.3810)],
)
def test_quiz_3_7(derivative: Sequence[NumberType], R: NumberType, pi: NumberType, premium: NumberType) -> None:
    T = len(derivative) - 1
    model = AssetOptionSolver(expire=T, W={T: derivative}, R=R, pi=pi, S={0: [0]})
    assert_almost_equal(model.derivative.premium, premium, 4)


@pytest.mark.parametrize(
    "derivative, state_price, premium",
    [([0, 7, 13], [0, 0.1, 0.1], 2), ([0, 10, 14], [0, 0.3, 0.2], 5.8)],
)
def test_quiz_3_8(derivative: Sequence[NumberType], state_price: Sequence[NumberType], premium: NumberType) -> None:
    result = np.array(derivative) @ np.array(state_price)
    assert_almost_equal(result, premium, 4)


@pytest.mark.parametrize(
    "K, T, S, u, R, type, premium",
    [
        (8, 2, 12, 1.3, 1.07, "call", 5.16),
        (9, 2, 11, 1.2, 1.06, "call", 3.167),
        (25, 2, 26, 1.4, 1.05, "put", 2.773),
        (24, 2, 26, 1.4, 1.08, "put", 2.004),
    ],
)
def test_quiz_3_9_10(K: NumberType, T: int, S: NumberType, u: NumberType, R: NumberType, type: OptionType, premium: NumberType) -> None:
    d = 1 / u
    model = AssetOptionSolver(expire=T, S=S, u=u, d=d, R=R, type=type, K=K)
    assert_almost_equal(model.derivative.premium, premium, 3)


def test_example_2_6() -> None:
    S = 100
    K = 100
    sigma_yr = 0.15
    r_yr = 0.05
    T_yr = 1
    C_0_yr = calculate_BS_Call(S, K, r_yr, T_yr, sigma_yr)
    sigma_mth = sigma_yr / math.sqrt(12)
    T_mth = T_yr * 12
    r_mth = r_yr / 12
    C_0_mth = calculate_BS_Call(S, K, r_mth, T_mth, sigma_mth)
    assert_almost_equal(C_0_mth, C_0_yr)
    assert_almost_equal(C_0_mth, 8.5916, 4)


def test_example_2_7() -> None:
    s_yr = 0.15
    s_mth = s_yr / math.sqrt(12)
    s_day = s_yr / math.sqrt(365)
    s_90_day = s_yr / math.sqrt(365 / 90)

    assert_almost_equal(s_mth, 0.0433, 4)
    assert_almost_equal(s_day, 0.00785, 5)
    assert_almost_equal(s_90_day, 0.0745, 4)

    s_mth = 0.05
    s_yr = 0.05 * math.sqrt(12)
    assert_almost_equal(s_yr, 0.1732, 4)


def test_example_2_8() -> None:
    S = 100
    K = 100
    sigma_yr = 0.15
    r_yr = 0.05
    T_yr = 1
    C_0_yr = calculate_BS_Put(S, K, r_yr, T_yr, sigma_yr)
    sigma_mth = sigma_yr / math.sqrt(12)
    T_mth = T_yr * 12
    r_mth = r_yr / 12
    C_0_mth = calculate_BS_Put(S, K, r_mth, T_mth, sigma_mth)
    assert_almost_equal(C_0_mth, C_0_yr)
    assert_almost_equal(C_0_mth, 3.71460076)


def test_example_2_9() -> None:
    S = 100
    K = 100
    T = 1
    sigma = 0.15
    r = 0.05
    N = 2
    u, d = calculate_BS_CRR_factor(sigma, T, N)
    R = calculate_BS_R(r, T, N)
    model = AssetOptionSolver(S=S, K=K, expire=2, R=R, u=u, d=d, type="call", style="european")
    assert_almost_equal(model.derivative.premium, 7.89449226, 4)


def test_example_2_10() -> None:
    S = 10
    K = 8
    T = 0.5
    r = 0.05
    C = 2.5

    sigma = calculate_BS_sigma(S, K, T, r, C, "call", [1e-6, 3.0])
    assert_almost_equal(sigma, 0.4248, 4)


def test_example_2_11() -> None:
    S = 39.540
    K = 40
    T = 0.21
    R = 1.045
    C = 1.660
    r = calculate_BS_r_from_R(R, 1, 1)

    sigma = calculate_BS_sigma(S, K, T, r, C, "call", [1e-6, 3.0])
    assert_almost_equal(sigma, 0.2358, 4)


def test_example_2_12() -> None:
    S = 20
    u = 1.1
    d = 1 / u
    K = 19

    R = {0: [1.03], 1: [1.035, 1.025], 2: [1.04, 1.03, 1.02], 3: [1.045, 1.035, 1.025, 1.015]}
    T = 4
    call_model = AssetOptionSolver(expire=T, S=S, u=u, d=d, K=K, type="call", R=R)
    call_premium = call_model.derivative.premium
    assert_almost_equal(call_premium, 3.2963, 4)

    put_model = AssetOptionSolver(expire=T, S=S, u=u, d=d, K=K, type="put", R=R)
    put_premium = put_model.derivative.premium
    assert_almost_equal(put_premium, 0.3083, 4)

    R_const = 1.03

    call_model = AssetOptionSolver(expire=T, S=S, u=u, d=d, K=K, type="call", R=R_const)
    call_premium = call_model.derivative.premium
    assert_almost_equal(call_premium, 3.4786, 4)

    put_model = AssetOptionSolver(expire=T, S=S, u=u, d=d, K=K, type="put", R=R_const)
    put_premium = put_model.derivative.premium
    assert_almost_equal(put_premium, 0.3599, 4)


def test_seminar_4_2() -> None:
    K = 24
    S = 20
    T_day = 90
    sigma_day = 0.05
    r_annual = 0.01

    # Part a
    r_day = r_annual / 365
    put_premium_day = calculate_BS_Put(S, K, r_day, T_day, sigma_day)
    call_premium_day = calculate_BS_Call(S, K, r_day, T_day, sigma_day)
    assert_almost_equal(call_premium_day, 2.435, 3)
    assert_almost_equal(put_premium_day, 6.376, 3)

    # Part b
    T_month = 90 / 30
    sigma_month = sigma_day * np.sqrt(30)
    r_month = r_annual / (365 / 30)
    put_premium_month = calculate_BS_Put(S, K, r_month, T_month, sigma_month)
    call_premium_month = calculate_BS_Call(S, K, r_month, T_month, sigma_month)
    assert_almost_equal(call_premium_month, 2.435, 3)
    assert_almost_equal(put_premium_month, 6.376, 3)

    # Part c
    expire = 2
    u, d = calculate_BS_CRR_factor(sigma_day, T_day, expire)
    R = calculate_BS_R(r_day, T_day, expire)
    call_model = AssetOptionSolver(expire=expire, K=K, S=20, u=u, d=d, type="call", R=R)
    assert_almost_equal(call_model.derivative.premium, 2.644, 3)

    expire = 2
    u, d = calculate_BS_CRR_factor(sigma_day, T_day, expire)
    R = calculate_BS_R(r_day, T_day, expire)
    put_model = AssetOptionSolver(expire=expire, K=K, S=20, u=u, d=d, type="put", R=R)
    assert_almost_equal(put_model.derivative.premium, 6.585, 3)

    # Part d
    expire = 10
    u, d = calculate_BS_CRR_factor(sigma_day, T_day, expire)
    R = calculate_BS_R(r_day, T_day, expire)
    call_model = AssetOptionSolver(expire=expire, K=K, S=20, u=u, d=d, type="call", R=R)
    assert_almost_equal(call_model.derivative.premium, 2.517, 3)

    expire = 10
    u, d = calculate_BS_CRR_factor(sigma_day, T_day, expire)
    R = calculate_BS_R(r_day, T_day, expire)
    put_model = AssetOptionSolver(expire=expire, K=K, S=20, u=u, d=d, type="put", R=R)
    assert_almost_equal(put_model.derivative.premium, 6.458, 3)


def test_workshop_4_1() -> None:
    K = 5
    S = 4
    u = 1.3
    d = 1 / u
    T_year = 2
    N = 8
    R = 1.1

    call = AssetOptionSolver(K=K, S=S, u=u, d=d, expire=N, R=R, type="call")
    assert_approx_equal(call.derivative.premium, 1.9568, 4)
    put = AssetOptionSolver(K=K, S=S, u=u, d=d, expire=N, R=R, type="put")
    assert_approx_equal(put.derivative.premium, 0.2894, 4)
    assert_approx_equal(put_call_parity(S, K, R, N), call.derivative.premium - put.derivative.premium)
    sigma = calculate_BS_sigma_from_u(u, T_year, N)
    assert_approx_equal(sigma, 0.5247, 4)
    r = calculate_BS_r_from_R(R, N, T_year)
    call_BS = calculate_BS_Call(S, K, r, T_year, sigma)
    assert_approx_equal(call_BS, 1.9650, 4)
    put_BS = calculate_BS_Put(S, K, r, T_year, sigma)
    assert_approx_equal(put_BS, 0.2975, 4)

    N = 6


def test_workshop_4_2() -> None:
    K = 11
    S = 10
    u = 1.2
    d = 1 / u
    T = 3
    R_states = {0: [1.02], 1: [1.03, 1.01], 2: [1.05, 1.02, 1.005]}
    call = AssetOptionSolver(expire=T, S=S, u=u, d=d, K=K, type="call", R=R_states)
    assert_almost_equal(call.derivative.premium, 1.0755, 4)
    put = AssetOptionSolver(expire=T, S=S, u=u, d=d, K=K, type="put", R=R_states)
    assert_approx_equal(put.derivative.premium, 1.4154, 4)
    pi = [[0.5091], [0.5364, 0.4818], [0.5909, 0.5091, 0.4682]]
    for i in range(T):
        assert_almost_equal(call.pi[i], pi[i], 4)


@pytest.mark.parametrize("sigma_year, sigma_month_pc", [(0.17, 4.907), (0.29, 8.372)])
def test_quiz_4_3(sigma_year: NumberType, sigma_month_pc: NumberType) -> None:
    sigma_month = sigma_year / math.sqrt(12)
    assert_almost_equal(sigma_month * 100, sigma_month_pc, 3)


@pytest.mark.parametrize("sigma_month, N, T_month, u_exp", [(0.16, 6, 12, 1.254), (0.20, 6, 12, 1.327)])
def test_quiz_4_4(sigma_month: NumberType, N: NumberType, T_month: NumberType, u_exp: NumberType) -> None:
    u, d = calculate_BS_CRR_factor(sigma_month, T_month, N)
    assert_almost_equal(u, u_exp, 3)


@pytest.mark.parametrize("S, K, T_year, r, sigma_year, call_exp", [(15, 10, 1, 0.03, 0.11, 5.296), (13, 11, 1, 0.03, 0.11, 2.344)])
def test_quiz_4_5(S: NumberType, K: NumberType, T_year: NumberType, r: NumberType, sigma_year: NumberType, call_exp: NumberType) -> None:
    C = calculate_BS_Call(S=S, K=K, r=r, T=T_year, sigma=sigma_year)
    assert_almost_equal(C, call_exp, 3)


@pytest.mark.parametrize("K, T_month, S, sigma_month, r_month, put_exp", [(57, 4, 50, 0.26, 0.05, 8.334), (57, 4, 50, 0.26, 0.05, 8.334)])
def test_quiz_4_6(K: NumberType, T_month: NumberType, S: NumberType, sigma_month: NumberType, r_month: NumberType, put_exp: NumberType) -> None:
    P = calculate_BS_Put(S=S, K=K, r=r_month, T=T_month, sigma=sigma_month)
    assert_almost_equal(P, put_exp, 3)


@pytest.mark.parametrize("r_year, T_month, R_exp", [(0.06, 3, 1.015), (0.07, 3, 1.018)])
def test_quiz_4_7(r_year: NumberType, T_month: NumberType, R_exp: NumberType) -> None:
    r_month = r_year / 12
    R = calculate_BS_R(r_month, T_month, 1)
    assert_almost_equal(R, R_exp, 3)


@pytest.mark.parametrize("P, K, S, r, C_exp", [(2.24, 24, 27, 0.05, 6.410), (2.04, 24, 26, 0.05, 5.210)])
def test_quiz_4_8(P: NumberType, K: NumberType, S: NumberType, r: NumberType, C_exp: NumberType) -> None:
    parity = put_call_parity(S, K, np.exp(r), 1)
    C = P + parity
    assert_almost_equal(C, C_exp, 2)


@pytest.mark.parametrize("K, T_month, S, sigma_year, r_year, call_exp", [(33, 8, 37, 0.34, 0.04, 6.728), (35, 8, 36, 0.31, 0.06, 4.818)])
def test_quiz_4_9(K: NumberType, T_month: NumberType, S: NumberType, sigma_year: NumberType, r_year: NumberType, call_exp: NumberType) -> None:
    r_month = r_year / 12
    sigma_month = sigma_year / np.sqrt(12)
    call = calculate_BS_Call(S, K, r_month, T_month, sigma_month)
    assert_almost_equal(call, call_exp, 2)
    T_year = T_month / 12
    call = calculate_BS_Call(S, K, r_year, T_year, sigma_year)
    assert_almost_equal(call, call_exp, 3)


@pytest.mark.parametrize("value, result", [(0.5, 0.6916), (0.15, 0.5597)])
def test_quiz_4_10(value: NumberType, result: NumberType) -> None:
    def approx(x: NumberType) -> NumberType:
        return float(1 / (1 + np.exp(-0.07056 * (x**3) - 1.5976 * x)))

    actual = approx(value)
    assert_almost_equal(actual, result, 4)


def test_example_3_1() -> None:
    K = 80
    S = 80
    u = 1.1
    d = 0.95
    R = 1.05
    N = 2
    eu_AssetOptionSolver = AssetOptionSolver(expire=N, S=S, u=u, d=d, R=R, type="call", style="european", K=K)
    american_AssetOptionSolver = AssetOptionSolver(expire=N, S=S, u=u, d=d, R=R, type="call", style="american", K=K)
    assert cast(ConstantPi, eu_AssetOptionSolver.pi).value == 2 / 3  # noqa: S101
    assert cast(ConstantPi, american_AssetOptionSolver.pi).value == 2 / 3  # noqa: S101

    # Check t = 2
    assert_almost_equal(eu_AssetOptionSolver.derivative[2], [0, 3.6, 16.8])
    assert_almost_equal(american_AssetOptionSolver.derivative[2], [0, 3.6, 16.8])

    # Check t = 1
    assert_almost_equal(eu_AssetOptionSolver.derivative[1], [2.29, 11.81], 2)
    assert_almost_equal(american_AssetOptionSolver.derivative[1], [2.29, 11.81], 2)

    # Check t = 0
    assert_almost_equal(eu_AssetOptionSolver.derivative[0], [8.22], 2)
    assert_almost_equal(american_AssetOptionSolver.derivative[0], [8.22], 2)


def test_example_3_2() -> None:
    K = 35
    S = 32
    sigma = 0.4
    r = 0.04
    T = 0.5
    N = 10
    u = math.exp(sigma * math.sqrt(T / N))
    d = 1 / u
    R = math.exp(r * T / N)

    # Get AssetOptionSolver obj
    eu_call = AssetOptionSolver(expire=N, S=S, u=u, d=d, R=R, type="call", style="european", K=K)
    eu_put = AssetOptionSolver(expire=N, S=S, u=u, d=d, R=R, type="put", style="european", K=K)
    us_call = AssetOptionSolver(expire=N, S=S, u=u, d=d, R=R, type="call", style="american", K=K)
    us_put = AssetOptionSolver(expire=N, S=S, u=u, d=d, R=R, type="put", style="american", K=K)

    # Check values
    eu_call_derivative = {
        10: [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 3.268279479, 10.76441295, 19.72891703, 30.44942164, 43.26989872],
        9: [0.0, 0.0, 0.0, 0.0, 0.0, 1.594443708, 6.918769276, 15.11627616, 24.91954129, 36.6431066],
        8: [0.0, 0.0, 0.0, 0.0, 0.7778559806, 4.188752334, 10.90413332, 19.8686374, 30.58914201],
        7: [0.0, 0.0, 0.0, 0.3794802685, 2.440321766, 7.456509686, 15.25571737, 25.0589825],
        6: [0.0, 0.0, 0.185131024, 1.384112144, 4.882614095, 11.24648741, 20.00780001],
        5: [0.0, 0.09031693839, 0.7696890475, 3.088105383, 7.977500633, 15.49825962],
        4: [0.04406149324, 0.4215709881, 1.899200239, 5.46724711, 11.63059506],
        3: [0.2281430322, 1.141596251, 3.636092109, 8.463138422],
        2: [0.6733191463, 2.356265498, 5.98372177],
        1: [1.493006216, 4.121227274],
        0: [2.772211739],
    }
    eu_put_derivative = {
        10: [21.91706497, 19.3543308, 16.28959871, 12.62453544, 8.241545898, 3.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        9: [20.62303414, 17.82051855, 14.46903574, 10.46105196, 5.667969358, 1.530443545, 0.0, 0.0, 0.0, 0.0],
        8: [19.21461043, 16.14987833, 12.48481507, 8.101825525, 3.638135608, 0.7807524815, 0.0, 0.0, 0.0],
        7: [17.68107734, 14.32959453, 10.32161075, 5.908008415, 2.236880391, 0.3982991985, 0.0, 0.0],
        6: [16.01071572, 12.34565246, 8.147793941, 4.105229163, 1.335451635, 0.2031914791, 0.0],
        5: [14.19070997, 10.27304313, 6.159332633, 2.745779447, 0.7804055847, 0.1036576959],
        4: [12.25110688, 8.245626836, 4.481710189, 1.781477581, 0.448692067],
        3: [10.27253909, 6.392909705, 3.155436041, 1.127713242],
        2: [8.359321246, 4.8007217, 2.159898492],
        1: [6.606541753, 3.502793289],
        0: [5.079165304],
    }
    us_call_derivative = {
        10: [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 3.268279479, 10.76441295, 19.72891703, 30.44942164, 43.26989872],
        9: [0.0, 0.0, 0.0, 0.0, 0.0, 1.594443708, 6.918769276, 15.11627616, 24.91954129, 36.6431066],
        8: [0.0, 0.0, 0.0, 0.0, 0.7778559806, 4.188752334, 10.90413332, 19.8686374, 30.58914201],
        7: [0.0, 0.0, 0.0, 0.3794802685, 2.440321766, 7.456509686, 15.25571737, 25.0589825],
        6: [0.0, 0.0, 0.185131024, 1.384112144, 4.882614095, 11.24648741, 20.00780001],
        5: [0.0, 0.09031693839, 0.7696890475, 3.088105383, 7.977500633, 15.49825962],
        4: [0.04406149324, 0.4215709881, 1.899200239, 5.46724711, 11.63059506],
        3: [0.2281430322, 1.141596251, 3.636092109, 8.463138422],
        2: [0.6733191463, 2.356265498, 5.98372177],
        1: [1.493006216, 4.121227274],
        0: [2.772211739],
    }
    us_put_derivative = {
        10: [21.91706497, 19.3543308, 16.28959871, 12.62453544, 8.241545898, 3.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        9: [20.69296419, 17.8904486, 14.53896579, 10.53098201, 5.737899405, 1.530443545, 0.0, 0.0, 0.0, 0.0],
        8: [19.3543308, 16.28959871, 12.62453544, 8.241545898, 3.67381027, 0.7807524815, 0.0, 0.0, 0.0],
        7: [17.8904486, 14.53896579, 10.53098201, 5.996690495, 2.255079743, 0.3982991985, 0.0, 0.0],
        6: [16.28959871, 12.62453544, 8.297868159, 4.159348765, 1.344735996, 0.2031914791, 0.0],
        5: [14.53896579, 10.53098201, 6.262295145, 2.77791786, 0.7851419813, 0.1036576959],
        4: [12.62453544, 8.42744449, 4.549915151, 1.800183593, 0.4511083295],
        3: [10.55174336, 6.518937664, 3.199356462, 1.138434857],
        2: [8.563239967, 4.886441351, 2.187534985],
        1: [6.752389137, 3.560005563],
        0: [5.181480278],
    }

    for i in range(N):
        assert_almost_equal(eu_call.derivative[i], eu_call_derivative[i], 6)
        assert_almost_equal(eu_put.derivative[i], eu_put_derivative[i], 6)
        assert_almost_equal(us_call.derivative[i], us_call_derivative[i], 6)
        assert_almost_equal(us_put.derivative[i], us_put_derivative[i], 6)


def test_seminar_5_2() -> None:
    S = 80
    u = 1.1
    d = 0.95
    K = 80
    R = 1.01
    T = 2
    eu_put = AssetOptionSolver(expire=T, S=S, u=u, d=d, K=K, type="put", style="european", R=R)
    us_put = AssetOptionSolver(expire=T, S=S, u=u, d=d, K=K, type="put", style="american", R=R)
    for i in range(T + 1):
        assert_almost_equal(eu_put.derivative[i], us_put.derivative[i])


def test_workshop_5_1() -> None:
    K = 11
    S = 10
    u = 1.2
    d = 1 / u
    T = 3
    R = {0: [1.02], 1: [1.03, 1.01], 2: [1.05, 1.02, 1.005]}
    us_call = AssetOptionSolver(expire=T, S=S, u=u, d=d, K=K, type="call", style="american", R=R)
    assert_almost_equal(us_call.derivative.premium, 1.0755, 4)
    us_put = AssetOptionSolver(expire=T, S=S, u=u, d=d, K=K, type="put", style="american", R=R)
    assert_almost_equal(us_put.derivative.premium, 1.6121, 4)


def test_quiz_5_5() -> None:
    S = 21
    u = 1.3
    d = 0.9
    K = 21
    T = 3
    R = 1.09
    us_put = AssetOptionSolver(expire=T, S=S, u=u, d=d, K=K, type="put", style="american", R=R)
    assert_almost_equal(us_put.derivative.premium, 1.011, 3)


def test_quiz_5_6() -> None:
    K = 32
    T = 2
    S = 31
    u = 1.3
    d = 1 / u
    R = {0: [1.06], 1: [1.07, 1.01]}
    us_put = AssetOptionSolver(expire=T, S=S, u=u, d=d, K=K, type="put", style="american", R=R)
    assert_almost_equal(us_put.derivative.premium, 3.758, 3)


@pytest.mark.parametrize(
    "S,u,d,K,B,R,T,type,style,barrier_type, premium",
    [
        (100, 1.1, 1 / 1.1, 60, 108, 1.08, 3, "call", "european", "up and in", 51.874),
        # Example 3.4
        (32, 1.093564, 1 / 1.093564, 35, 50, 1.002002, 10, "call", "european", "up and out", 1.302),
        (32, 1.093564, 1 / 1.093564, 35, 50, 1.002002, 10, "call", "american", "up and out", 2.689),
        (32, 1.093564, 1 / 1.093564, 35, 50, 1.002002, 10, "put", "european", "up and out", 5.076),
        (32, 1.093564, 1 / 1.093564, 35, 50, 1.002002, 10, "put", "american", "up and out", 5.179),
        # Workshop 6.1-6.5
        (80, 1.1, 0.95, 80, 85, 1.05, 2, "call", "european", "up and out", 0.726),
        (80, 1.1, 0.95, 80, 85, 1.05, 2, "call", "european", "up and in", 7.498),
        (80, 1.1, 0.95, 80, 85, 1.05, 2, "call", "european", "down and out", 0.000),
        (80, 1.1, 0.95, 80, 85, 1.05, 2, "call", "european", "down and in", 8.224),
        (80, 1.1, 0.95, 80, 85, 1.05, 2, "call", "american", "up and out", 0.726),
        (80, 1.1, 0.95, 80, 85, 1.05, 2, "call", "american", "up and in", 7.498),
        (80, 1.1, 0.95, 80, 85, 1.05, 2, "call", "american", "down and out", 0.000),
        (80, 1.1, 0.95, 80, 85, 1.05, 2, "call", "american", "down and in", 8.224),
        # Workshop 6.1
        (20, 1.2, 1 / 1.2, 20, 10, 1.1, 4, "call", "european", "up and in", 6.625),
        (20, 1.2, 1 / 1.2, 20, 30, 1.1, 4, "call", "european", "up and in", 4.733),
        (20, 1.2, 1 / 1.2, 20, 30, 1.1, 4, "call", "european", "up and out", 1.892),
        (20, 1.2, 1 / 1.2, 20, 10, 1.1, 4, "call", "american", "up and in", 6.625),
        (20, 1.2, 1 / 1.2, 20, 30, 1.1, 4, "call", "american", "up and in", 4.733),
        (20, 1.2, 1 / 1.2, 20, 30, 1.1, 4, "call", "american", "up and out", 5.108),
    ],
)
def test_barrier(
    S: NumberType,
    u: NumberType,
    d: NumberType,
    K: NumberType,
    B: NumberType,
    R: NumberType,
    T: int,
    type: OptionType,
    style: OptionStyle,
    barrier_type: BarrierType,
    premium: NumberType,
) -> None:
    model = BarrierAssetOptionSolver(expire=T, S=S, u=u, d=d, K=K, type=type, style=style, R=R, B=B, barrier_type=barrier_type)
    assert_almost_equal(model.derivative.premium, premium, 3)


def test_quiz_6_4() -> None:
    barrier_type: BarrierType = "down and in"
    K = 19
    T = 2
    B = 17
    S = 20
    u = 1.2
    d = 0.8
    R = 1.1
    model = BarrierAssetOptionSolver(expire=T, S=S, u=u, d=d, K=K, type="call", style="american", R=R, B=B, barrier_type=barrier_type)
    assert_almost_equal(model.derivative.premium, 0.03099, 4)


@pytest.mark.parametrize(
    "K, B, S, u, d, R, premium",
    [
        (11, 13.25, 11, 1.2, 0.6, 1.06, 1.5912),
        (11, 13.25, 11, 1.2, 0.6, 1.08, 1.6296),
    ],
)
def test_quiz_6_5(
    K: NumberType,
    B: NumberType,
    S: NumberType,
    u: NumberType,
    d: NumberType,
    R: NumberType,
    premium: NumberType,
) -> None:
    barrier_type: BarrierType = "up and out"
    T = 2
    model = BarrierAssetOptionSolver(expire=T, S=S, u=u, d=d, K=K, type="call", style="american", R=R, B=B, barrier_type=barrier_type)
    assert_almost_equal(model.derivative.premium, premium, 4)


@pytest.mark.parametrize("vanilla, S, B, down_in", [(1.41, 38, 45, 1.41), (1.83, 38, 48, 1.83)])
def test_quiz_6_6(vanilla: NumberType, S: NumberType, B: NumberType, down_in: NumberType) -> None:
    assert_almost_equal(vanilla, down_in, 4)


@pytest.mark.parametrize(
    "P_upin, P_upout, K, S, R, C",
    [
        (2.25, 1.2, 21, 25, 1.01, 7.6579),
        (2.67, 1.6, 25, 23, 1.04, 3.2315),
    ],
)
def test_quiz_6_7(P_upin: NumberType, P_upout: NumberType, K: NumberType, S: NumberType, R: NumberType, C: NumberType) -> None:
    P_vanilla = P_upin + P_upout
    parity = S - K / R
    C_vanilla = P_vanilla + parity
    assert_almost_equal(C_vanilla, C, 4)


@pytest.mark.parametrize(
    "K, S, u, d, R, premium",
    [
        (24, 24, 1.5, 0.8, 1.02, 3.8425),
        (22, 22, 1.5, 0.8, 1.05, 4.0978),
    ],
)
def test_quiz_6_10(
    K: NumberType,
    S: NumberType,
    u: NumberType,
    d: NumberType,
    R: NumberType,
    premium: NumberType,
) -> None:
    barrier_type: BarrierType = "down and out"
    B = K - 1
    T = 2
    model = BarrierAssetOptionSolver(expire=T, S=S, u=u, d=d, K=K, type="call", style="european", R=R, B=B, barrier_type=barrier_type)
    assert_almost_equal(model.derivative.premium, premium, 4)


def test_seminar_7_4() -> None:
    k = 1.3
    F = 500000
    X = 1.3
    u = 1.1
    d = 0.9
    Rd = 1.01
    Rf = 1.02
    expire = 2
    model = ForexOptionSolver(expire=expire, X=X, F=F, Rf=Rf, Rd=Rd, k=k, u=u, d=d, type="put", style="european")
    assert_almost_equal(model.derivative.premium, 39648, 0)


def test_quiz_7_5() -> None:
    F = 1000
    k = 1.3
    X = 1.2
    u = 1.3
    d = 0.7
    T = 1
    Rd = 1.2
    Rf = 1.1
    model = ForexOptionSolver(expire=T, X=X, F=F, Rf=Rf, Rd=Rd, k=k, u=u, d=d, type="call", style="european")
    assert_almost_equal(model.derivative.premium, 141.2, 1)


def test_quiz_7_7() -> None:
    F = 100
    k = 1.04
    X = 1.08
    u = 1.6
    d = 0.9
    T = 2
    Rd = 1.4
    Rf = 1.1
    model = ForexOptionSolver(expire=T, X=X, F=F, Rf=Rf, Rd=Rd, k=k, u=u, d=d, type="put", style="american")
    assert_almost_equal(model.derivative.premium, 2.271, 3)


def test_quiz_7_2() -> None:
    X_0 = 1.3
    Rd = 1.2
    Rf = 1.1
    F = 200
    exercised_value = calculate_forward_rate_forex(Rd, Rf, X_0) * F
    assert_almost_equal(exercised_value, 283.64, 2)


def test_quiz_7_9() -> None:
    X_0 = 1 / 0.66
    Rd = 1.08
    Rf = 1.03
    forward_rate = calculate_forward_rate_forex(Rd, Rf, X_0)
    assert_almost_equal(forward_rate, 1.589, 3)


def test_quiz_7_10() -> None:
    S_11 = 21
    S_10 = 12
    pi = 0.5
    R = 1.02
    S_0 = 1 / R * (pi * S_11 + (1 - pi) * S_10)
    forward_rate = calculate_forward_rate(R, S_0)
    assert_almost_equal(forward_rate, 16.5, 1)
