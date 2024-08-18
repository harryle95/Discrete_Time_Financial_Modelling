import math
from collections.abc import Sequence

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
    present_value,
    put_call_parity,
)
from src.models import (
    OptionType,
    asset_factory,
    base_derivative_factory,
    derivative_factory,
    interest_factory,
    pi_factory,
)
from src.solver import OneStepSolver, Solver


@pytest.mark.parametrize(
    "strike, asset, type, exp_value",
    [
        (22, 35, "call", 13),  # Quiz 1-7
        (14, 5, "put", 9),  # Quiz 1-8
    ],
)
def test_quiz_1_7(
    strike: float,
    asset: float,
    type: OptionType,
    exp_value: float,
) -> None:
    T = 1
    W = base_derivative_factory(strike=strike, expire=T, type=type)
    value = W.value(T, asset)
    assert_approx_equal(value, exp_value)


def test_quiz_1_9() -> None:
    N = 100
    P_0 = 1.47
    K = 14
    T = 1
    R = 1.03
    S_1 = 7
    W = base_derivative_factory(strike=K, expire=T, type="put")
    value = W.value(T, S_1)
    assert_approx_equal((value - P_0 * R) * N, 548.59)


def test_quiz_1_10() -> None:
    N = 100
    K = 42
    C_0 = 0.55
    T = 1
    R = 1.03
    S_1 = 36
    W = base_derivative_factory(strike=K, expire=T, type="call")
    value = W.value(T, S_1)
    profit = C_0 * N * R - N * value
    assert_approx_equal(profit, 56.65)


def test_example_1_13() -> None:
    R = 10 / 9
    S_11 = 20 / 3
    W_11 = 7
    S_10 = 40 / 9
    W_10 = 2
    S_0 = 5
    T = 1
    model = OneStepSolver(
        expire=T,
        asset_states={0: [S_0], 1: [S_10, S_11]},
        derivative_states={1: [W_10, W_11]},
        interest_value=R,
    )
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
    T = 1
    model = OneStepSolver(expire=T, strike=K, interest_value=R, asset_states={0: [S_0], 1: [S_10, S_11]}, type="call")
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
    T = 1
    model = OneStepSolver(
        expire=T, interest_value=R, asset_states={0: [S_0], 1: [S_10, S_11]}, derivative_states={1: [W_10, W_11]}
    )

    p_up = model.pi[0, 0]
    p_down = 1 - p_up
    W0 = model.premium
    assert_approx_equal(p_up, 1 / 2)
    assert_approx_equal(p_down, 1 / 2)
    assert_approx_equal(W0, 4.05)


def test_workshop_2_1() -> None:
    S_0 = 4
    S_11 = 8
    S_10 = 2
    r = 0.25
    K = 6
    T = 1
    R = 1 + r

    # Call Model
    call_model = OneStepSolver(
        expire=T,
        asset_states={0: [S_0], 1: [S_10, S_11]},
        interest_value=R,
        strike=6,
        type="call",
    )
    C_0_replicate = call_model.premium_replicating
    C_0_general = call_model.premium
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
    put_model = OneStepSolver(
        expire=T,
        asset_states={0: [S_0], 1: [S_10, S_11]},
        interest_value=R,
        strike=6,
        type="put",
    )
    P_0 = put_model.premium
    assert_approx_equal(P_0, 1.6)
    # Parity
    actual = C_0_general - P_0
    expected = put_call_parity(S_0, K, R, T)
    assert_approx_equal(actual, -0.8)
    assert_approx_equal(expected, -0.8)


@pytest.mark.parametrize("W_11, S_11, W_10, S_10, R, T, H0, H1", [(0, 50, 5, 10, 1.25, 1, 5, -0.125)])
def test_quiz_2_1(W_11: float, S_11: float, W_10: float, S_10: float, R: float, T: int, H0: float, H1: float) -> None:
    model = OneStepSolver(
        expire=T, asset_states={0: [0], 1: [S_10, S_11]}, derivative_states={1: [W_10, W_11]}, interest_value=R
    )
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
        (18, 1.18, 0.98, 1.08, 1, 0.5),
        (12, 1.11, 0.94, 1.08, 1, 0.8235),
    ],
)
def test_quiz_2_3(S_0: float, u: float, d: float, R: float, T: int, pi: float) -> None:
    asset = asset_factory(T, S_0, u, d)
    interest = interest_factory(value=R)
    pi_model = pi_factory(asset=asset, R=interest)
    assert_approx_equal(pi_model[0, 0], pi, 4)


def test_quiz_2_4() -> None:
    S_0 = 20
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
    K: float, S_11: float, S_10: float, R: float, pi: float, T: int, type: OptionType, premium: float
) -> None:
    model = OneStepSolver(
        expire=T, strike=K, asset_states={0: [0], 1: [S_10, S_11]}, interest_value=R, pi_value=pi, type=type
    )
    assert_approx_equal(model.premium, premium, 4)


@pytest.mark.parametrize("K, C_0, S_0, R, premium", [(32, 2, 26, 1.02, 7.373), (34, 2.65, 27, 1.08, 7.131)])
def test_quiz_2_7(K: float, C_0: float, S_0: float, R: float, premium: float) -> None:
    parity = put_call_parity(S_0, K, R, 1)
    P_0 = C_0 - parity
    assert_approx_equal(P_0, premium, 4)


@pytest.mark.parametrize(
    "S_0, u, R, profit",
    [
        (17, 1.07, 1.02, 1.452),
        (18, 1.08, 1.03, 1.873),
    ],
)
def test_quiz_2_8(S_0: float, u: float, R: float, profit: float) -> None:
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
    W_0: float, S_0: float, W_11: float, W_10: float, S_11: float, S_10: float, N: int, R: float, H1: float
) -> None:
    model = OneStepSolver(
        expire=1,
        asset_states={0: [S_0], 1: [S_10, S_11]},
        derivative_states={0: [W_0], 1: [W_10 * N, W_11 * N]},
        interest_value=R,
    )

    assert_approx_equal(model.H1, H1, 3)


@pytest.mark.parametrize(
    "S, u, d, T, result",
    [(4, 2, 1 / 2, 3, [[4.0], [2.0, 8.0], [1.0, 4.0, 16.0], [1 / 2, 2, 8, 32]])],
)
def test_example_2_1(S: float, u: float, d: float, T: int, result: Sequence[float]) -> None:
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
    model = Solver(expire=expire, strike=K, u=u, d=d, S=S, interest_value=R, type="call")
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
    model = Solver(expire=expire, strike=K, u=u, d=d, S=S, interest_value=R, type="call")
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
    model = Solver(expire=expire, strike=K, u=u, d=d, S=S, interest_value=R, type="call")
    derivative = model.derivative
    assert_almost_equal(model.state_price[3], [8 / 125, 24 / 125, 24 / 125, 8 / 125])
    assert_almost_equal(model.premium, derivative[0, 0])


def test_workshop_3_1() -> None:
    K = 23
    S = 20
    u = 1.06
    d = 0.95
    R = 1.01
    T = 10
    put_model = Solver(expire=T, strike=K, S=S, u=u, d=d, interest_value=R, type="put")
    put_premium = put_model.premium

    call_model = Solver(expire=T, strike=K, S=S, u=u, d=d, interest_value=R, type="call")
    call_premium = call_model.premium
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
    put_model = Solver(expire=T, strike=K, S=S, u=u, d=d, interest_value=R, type="put")
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
    assert_almost_equal(put_model.premium, 0.4793, 4)
    assert_almost_equal(put_model.derivative[0, 0], 0.4793, 4)


@pytest.mark.parametrize("S, u, d, T, largest, smallest", [(12, 1.2, 0.9, 3, 20.736, 8.748)])
def test_quiz_3_3(S: float, u: float, d: float, T: int, largest: float, smallest: float) -> None:
    asset = asset_factory(T, S, u, d)
    assert_almost_equal(asset[T, T], largest)
    assert_almost_equal(asset[T, 0], smallest)


@pytest.mark.parametrize(
    "C, pi, R, T, premium",
    [([0, 0, 0, 3.84], 0.3, 1.13, 3, 0.0719), ([0, 0, 0, 3.5], 0.3, 1.18, 3, 0.0575)],
)
def test_quiz_3_4(C: Sequence[float], pi: float, R: float, T: int, premium: float) -> None:
    model = Solver(expire=T, derivative_states={T: C}, pi_value=pi, interest_value=R, asset_states={0: [0]})
    assert_almost_equal(model.premium, premium, 4)


@pytest.mark.parametrize("K, T, asset, put", [(30, 3, [10, 15, 25, 35], [20, 15, 5, 0])])
def test_quiz_3_5(K: float, T: int, asset: Sequence[float], put: float) -> None:
    asset_model = asset_factory(states={0: [0], T: asset}, steps=T)
    pi = pi_factory(value=0.1)
    R = interest_factory(value=0.1)
    derivative = derivative_factory(expire=T, strike=K, type="put", pi=pi, R=R, asset=asset_model)
    assert_almost_equal(derivative.final, put)


@pytest.mark.parametrize("R, state_22, result", [(1.2, 0.27, 0.0984), (1.3, 0.12, 0.1788)])
def test_quiz_3_6(R: float, state_22: float, result: float) -> None:
    pi = math.sqrt(state_22 * R**2)
    pi_down = 1 - pi
    state_20 = (pi_down) ** 2 / R**2
    assert_almost_equal(float(state_20), result, 4)


@pytest.mark.parametrize(
    "derivative, R, pi, premium",
    [([0, 0, 1, 0], 1.06, 0.5, 0.3149), ([0, 0, 1, 0], 1.05, 0.7, 0.3810)],
)
def test_quiz_3_7(derivative: Sequence[float], R: float, pi: float, premium: float) -> None:
    T = len(derivative) - 1
    model = Solver(expire=T, derivative_states={T: derivative}, interest_value=R, pi_value=pi, asset_states={0: [0]})
    assert_almost_equal(model.premium, premium, 4)


@pytest.mark.parametrize(
    "derivative, state_price, premium",
    [([0, 7, 13], [0, 0.1, 0.1], 2), ([0, 10, 14], [0, 0.3, 0.2], 5.8)],
)
def test_quiz_3_8(derivative: Sequence[float], state_price: Sequence[float], premium: float) -> None:
    result = np.array(derivative) @ np.array(state_price)
    assert_almost_equal(float(result), premium, 4)


@pytest.mark.parametrize(
    "K, T, S, u, R, type, premium",
    [
        (8, 2, 12, 1.3, 1.07, "call", 5.16),
        (9, 2, 11, 1.2, 1.06, "call", 3.167),
        (25, 2, 26, 1.4, 1.05, "put", 2.773),
        (24, 2, 26, 1.4, 1.08, "put", 2.004),
    ],
)
def test_quiz_3_9_10(K: float, T: int, S: float, u: float, R: float, type: OptionType, premium: float) -> None:
    d = 1 / u
    model = Solver(expire=T, S=S, u=u, d=d, interest_value=R, type=type, strike=K)
    assert_almost_equal(model.premium, premium, 3)


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
    model = Solver(S=S, strike=K, expire=2, interest_value=R, u=u, d=d, type="call")
    assert_almost_equal(model.premium, 7.89449226, 4)


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
    call_model = Solver(expire=T, S=S, u=u, d=d, strike=K, type="call", interest_states=R)
    call_premium = call_model.premium
    assert_almost_equal(call_premium, 3.2963, 4)

    put_model = Solver(expire=T, S=S, u=u, d=d, strike=K, type="put", interest_states=R)
    put_premium = put_model.premium
    assert_almost_equal(put_premium, 0.3083, 4)

    R_const = 1.03

    call_model = Solver(expire=T, S=S, u=u, d=d, strike=K, type="call", interest_value=R_const)
    call_premium = call_model.premium
    assert_almost_equal(call_premium, 3.4786, 4)

    put_model = Solver(expire=T, S=S, u=u, d=d, strike=K, type="put", interest_value=R_const)
    put_premium = put_model.premium
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
    call_model = Solver(expire=expire, strike=K, S=20, u=u, d=d, type="call", interest_value=R)
    assert_almost_equal(call_model.premium, 2.644, 3)

    expire = 2
    u, d = calculate_BS_CRR_factor(sigma_day, T_day, expire)
    R = calculate_BS_R(r_day, T_day, expire)
    put_model = Solver(expire=expire, strike=K, S=20, u=u, d=d, type="put", interest_value=R)
    assert_almost_equal(put_model.premium, 6.585, 3)

    # Part d
    expire = 10
    u, d = calculate_BS_CRR_factor(sigma_day, T_day, expire)
    R = calculate_BS_R(r_day, T_day, expire)
    call_model = Solver(expire=expire, strike=K, S=20, u=u, d=d, type="call", interest_value=R)
    assert_almost_equal(call_model.premium, 2.517, 3)

    expire = 10
    u, d = calculate_BS_CRR_factor(sigma_day, T_day, expire)
    R = calculate_BS_R(r_day, T_day, expire)
    put_model = Solver(expire=expire, strike=K, S=20, u=u, d=d, type="put", interest_value=R)
    assert_almost_equal(put_model.premium, 6.458, 3)


def test_workshop_4_1() -> None:
    K = 5
    S = 4
    u = 1.3
    d = 1 / u
    T_year = 2
    N = 8
    R = 1.1

    call = Solver(strike=K, S=S, u=u, d=d, expire=N, interest_value=R, type="call")
    assert_approx_equal(call.premium, 1.9568, 4)
    put = Solver(strike=K, S=S, u=u, d=d, expire=N, interest_value=R, type="put")
    assert_approx_equal(put.premium, 0.2894, 4)
    assert_approx_equal(put_call_parity(S, K, R, N), call.premium - put.premium)
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
    call = Solver(expire=T, S=S, u=u, d=d, strike=K, type="call", interest_states=R_states)
    assert_almost_equal(call.premium, 1.0755, 4)
    put = Solver(expire=T, S=S, u=u, d=d, strike=K, type="put", interest_states=R_states)
    assert_approx_equal(put.premium, 1.4154, 4)
    pi = [[0.5091], [0.5364, 0.4818], [0.5909, 0.5091, 0.4682]]
    for i in range(T):
        assert_almost_equal(call.pi[i], pi[i], 4)


@pytest.mark.parametrize("sigma_year, sigma_month_pc", [(0.17, 4.907), (0.29, 8.372)])
def test_quiz_4_3(sigma_year: float, sigma_month_pc: float) -> None:
    sigma_month = sigma_year / math.sqrt(12)
    assert_almost_equal(sigma_month * 100, sigma_month_pc, 3)


@pytest.mark.parametrize("sigma_month, N, T_month, u_exp", [(0.16, 6, 12, 1.254), (0.20, 6, 12, 1.327)])
def test_quiz_4_4(sigma_month: float, N: int, T_month: int, u_exp: float) -> None:
    u, d = calculate_BS_CRR_factor(sigma_month, T_month, N)
    assert_almost_equal(u, u_exp, 3)


@pytest.mark.parametrize(
    "S, K, T_year, r, sigma_year, call_exp", [(15, 10, 1, 0.03, 0.11, 5.296), (13, 11, 1, 0.03, 0.11, 2.344)]
)
def test_quiz_4_5(S: float, K: float, T_year: int, r: float, sigma_year: float, call_exp: float) -> None:
    C = calculate_BS_Call(S=S, K=K, r=r, T=T_year, sigma=sigma_year)
    assert_almost_equal(C, call_exp, 3)


@pytest.mark.parametrize(
    "K, T_month, S, sigma_month, r_month, put_exp", [(57, 4, 50, 0.26, 0.05, 8.334), (57, 4, 50, 0.26, 0.05, 8.334)]
)
def test_quiz_4_6(K: float, T_month: int, S: float, sigma_month: float, r_month: float, put_exp: float) -> None:
    P = calculate_BS_Put(S=S, K=K, r=r_month, T=T_month, sigma=sigma_month)
    assert_almost_equal(P, put_exp, 3)


@pytest.mark.parametrize("r_year, T_month, R_exp", [(0.06, 3, 1.015), (0.07, 3, 1.018)])
def test_quiz_4_7(r_year: float, T_month: int, R_exp: float) -> None:
    r_month = r_year / 12
    R = calculate_BS_R(r_month, T_month, 1)
    assert_almost_equal(R, R_exp, 3)


@pytest.mark.parametrize("P, K, S, r, C_exp", [(2.24, 24, 27, 0.05, 6.410), (2.04, 24, 26, 0.05, 5.210)])
def test_quiz_4_8(P: float, K: float, S: float, r: float, C_exp: float) -> None:
    parity = put_call_parity(S, K, np.exp(r), 1)
    C = P + parity
    assert_almost_equal(C, C_exp, 2)


@pytest.mark.parametrize(
    "K, T_month, S, sigma_year, r_year, call_exp", [(33, 8, 37, 0.34, 0.04, 6.728), (35, 8, 36, 0.31, 0.06, 4.818)]
)
def test_quiz_4_9(K: float, T_month: int, S: float, sigma_year: float, r_year: float, call_exp: float) -> None:
    r_month = r_year / 12
    sigma_month = sigma_year / np.sqrt(12)
    call = calculate_BS_Call(S, K, r_month, T_month, sigma_month)
    assert_almost_equal(call, call_exp, 2)
    T_year = T_month / 12
    call = calculate_BS_Call(S, K, r_year, T_year, sigma_year)
    assert_almost_equal(call, call_exp, 3)


@pytest.mark.parametrize("value, result", [(0.5, 0.6916), (0.15, 0.5597)])
def test_quiz_4_10(value: float, result: float) -> None:
    def approx(x: float) -> float:
        return float(1 / (1 + np.exp(-0.07056 * (x**3) - 1.5976 * x)))

    actual = approx(value)
    assert_almost_equal(actual, result, 4)
