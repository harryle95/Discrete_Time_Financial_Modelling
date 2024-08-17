import numpy as np
import pytest

from src.helpers import present_value, put_call_parity
from src.models import *
from src.multi_steps import MultiStepModel
from src.one_step import OneStepModel


def test_quiz_1_7() -> None:
    K = 22
    S_1 = 35
    T = 1
    W = base_derivative_factory(DerivativeParams(strike=K, expire=T, type="call"))
    value = W.value(T, S_1)
    assert value == 13


def test_quiz_1_8() -> None:
    K = 14
    S_1 = 5
    T = 1
    W = base_derivative_factory(DerivativeParams(strike=K, expire=T, type="put"))
    value = W.value(T, S_1)
    assert value == 9


def test_quiz_1_9() -> None:
    N = 100
    P_0 = 1.47
    K = 14
    T = 1
    R = 1.03
    S_1 = 7
    W = base_derivative_factory(DerivativeParams(strike=K, expire=T, type="put"))
    value = W.value(T, S_1)
    assert (value - P_0 * R) * N == 548.59


def test_quiz_1_10() -> None:
    N = 100
    K = 42
    C_0 = 0.55
    T = 1
    R = 1.03
    S_1 = 36
    W = base_derivative_factory(DerivativeParams(strike=K, expire=T, type="call"))
    value = W.value(T, S_1)
    profit = C_0 * N * R - N * value
    assert profit == pytest.approx(56.65, 1e-2)


def test_example_1_13() -> None:
    R = 10 / 9
    S_11 = 20 / 3
    W_11 = 7
    S_10 = 40 / 9
    W_10 = 2
    S_0 = 5
    model = OneStepModel(
        asset_params=TerminalAssetParams(
            [S_10, S_11],
            steps=1,
            S_0=S_0,
        ),
        derivative_params=TerminalDerivativeParams(
            V_T=[W_10, W_11],
            expire=1,
        ),
        R=R,
    )
    H0 = model.H0
    H1 = model.H1
    W0 = model.premium_replicating
    assert pytest.approx(-7.2) == H0
    assert pytest.approx(2.25) == H1
    assert pytest.approx(4.05) == W0


def test_example_1_14() -> None:
    K = 4
    R = 6 / 5
    S_11 = 6
    S_10 = 3
    S_0 = 4
    T = 1
    model = OneStepModel(
        asset_params=TerminalAssetParams(V_T=[S_10, S_11], steps=T, S_0=S_0),
        derivative_params=DerivativeParams(strike=K, expire=T, type="call"),
        R=R,
    )
    H0 = model.H0
    H1 = model.H1
    W0 = model.premium_replicating
    assert pytest.approx(-5 / 3) == H0
    assert pytest.approx(2 / 3) == H1
    assert pytest.approx(1) == W0


def test_example_1_18() -> None:
    R = 10 / 9
    S_11 = 20 / 3
    W_11 = 7
    S_10 = 40 / 9
    W_10 = 2
    S_0 = 5
    model = OneStepModel(
        asset_params=TerminalAssetParams(V_T=[S_10, S_11], steps=1, S_0=S_0),
        derivative_params=TerminalDerivativeParams(V_T=[W_10, W_11], expire=1),
        R=R,
    )
    p_up = model.pi.p_up
    p_down = model.pi.p_down
    W0 = model.premium
    assert p_up == pytest.approx(1 / 2, 1e-3)
    assert p_down == pytest.approx(1 / 2, 1e-3)
    assert pytest.approx(4.05) == W0


def test_workshop_2_1() -> None:
    S_0 = 4
    S_11 = 8
    S_10 = 2
    r = 0.25
    K = 6
    T = 1
    R = 1 + r

    # Call Model
    call_model = OneStepModel(
        asset_params=TerminalAssetParams(V_T=[S_10, S_11], steps=T, S_0=S_0),
        derivative_params=DerivativeParams(strike=K, expire=T, type="call"),
        R=R,
    )
    C_0_replicate = call_model.premium_replicating
    C_0_general = call_model.premium
    H0 = call_model.H0
    H1 = call_model.H1
    call_p_up = call_model.pi.p_up
    call_p_down = call_model.pi.p_down

    assert pytest.approx(-8 / 15) == H0
    assert pytest.approx(1 / 3) == H1
    assert C_0_replicate == pytest.approx(0.8)

    assert call_p_up == pytest.approx(1 / 2)
    assert call_p_down == pytest.approx(1 / 2)
    assert C_0_general == pytest.approx(0.8)

    # Put Model
    put_model = OneStepModel(
        asset_params=TerminalAssetParams(V_T=[S_10, S_11], steps=T, S_0=S_0),
        derivative_params=DerivativeParams(strike=K, expire=T, type="put"),
        R=R,
    )
    P_0 = put_model.premium
    put_p_up = put_model.pi.p_up
    put_p_down = put_model.pi.p_down
    assert put_p_up == pytest.approx(1 / 2)
    assert put_p_down == pytest.approx(1 / 2)
    assert pytest.approx(1.6) == P_0

    # Parity
    actual = C_0_general - P_0
    expected = put_call_parity(S_0, K, R, T)
    assert actual == pytest.approx(-0.8)
    assert expected == pytest.approx(-0.8)


@pytest.mark.parametrize("W_11, S_11, W_10, S_10, R, T, H0, H1", [(0, 50, 5, 10, 1.25, 1, 5, -0.125)])
def test_quiz_2_1(W_11, S_11, W_10, S_10, R, T, H0, H1) -> None:
    model = OneStepModel(
        asset_params=TerminalAssetParams(V_T=[S_10, S_11], steps=T, S_0=0),
        derivative_params=TerminalDerivativeParams(V_T=[W_10, W_11], expire=T),
        R=R,
    )
    assert pytest.approx(H0) == model.H0
    assert pytest.approx(H1) == model.H1


def test_quiz_2_2() -> None:
    R = 1.1
    future = ((109, 1), (134, 3), (100, 0), (122, 2))
    values = {fv: present_value(fv[0], R, fv[1]) for fv in future}
    s_values = sorted(values.items(), key=lambda item: item[1], reverse=True)
    assert s_values[0][0] == (122, 2)


@pytest.mark.parametrize(
    "S_0, u, d, R, T, pi",
    [
        (18, 1.18, 0.98, 1.08, 1, 0.5),
        (12, 1.11, 0.94, 1.08, 1, 0.8235),
    ],
)
def test_quiz_2_3(S_0, u, d, R, T, pi) -> None:
    model = OneStepModel(
        asset_params=CRRAssetParams(S_0=S_0, u=u, d=d, steps=T),
        derivative_params=TerminalDerivativeParams([0, 0], T),
        R=R,
    )
    assert model.pi.p_up == pytest.approx(pi, 1e-4)


def test_quiz_2_4() -> None:
    S_0 = 20
    u = 1.1
    d = 0.8
    model = asset_factory(CRRAssetParams(S_0, u, d, 1))
    assert model[1, 1] == pytest.approx(22)
    assert model[1, 0] == pytest.approx(16)


@pytest.mark.parametrize(
    "K, S_11, S_10, R, pi, T, premium",
    [
        (15, 24, 11, 1.05, 0.64, 1, 1.371),
        (18, 25, 12, 1.05, 0.68, 1, 1.829),
    ],
)
def test_quiz_2_5(K, S_11, S_10, R, pi, T, premium) -> None:
    model = OneStepModel(
        TerminalAssetParams([S_10, S_11], T, 0),
        DerivativeParams(K, T, type="put"),
        R=R,
        pi_values=TerminalPiParams(R=R, p_up=pi, p_down=1 - pi),
    )
    assert model.premium == pytest.approx(premium, 1e-3)


@pytest.mark.parametrize(
    "K, S_11, S_10, R, pi, T, premium",
    [(21, 34, 15, 1.02, 0.48, 1, 6.118), [20, 38, 19, 1.05, 0.48, 1, 8.229]],
)
def test_quiz_2_6(K, S_11, S_10, R, pi, T, premium) -> None:
    model = OneStepModel(
        TerminalAssetParams([S_10, S_11], T, 0),
        DerivativeParams(K, T, type="call"),
        R=R,
        pi_values=TerminalPiParams(R=R, p_up=pi, p_down=1 - pi),
    )
    assert model.premium == pytest.approx(premium, 1e-3)


@pytest.mark.parametrize("K, C_0, S_0, R, premium", [(32, 2, 26, 1.02, 7.373), (34, 2.65, 27, 1.08, 7.131)])
def test_quiz_2_7(K, C_0, S_0, R, premium) -> None:
    parity = put_call_parity(S_0, K, R, 1)
    P_0 = C_0 - parity
    assert pytest.approx(premium, 1e-3) == P_0


@pytest.mark.parametrize(
    "S_0, u, R, profit",
    [
        (17, 1.07, 1.02, 1.452),
        (18, 1.08, 1.03, 1.873),
    ],
)
def test_quiz_2_8(S_0, u, R, profit) -> None:
    d = 1 / u
    model = asset_factory(CRRAssetParams(S_0, u, d, 1))
    S_10 = model[1, 0]
    profit = S_0 * R - S_10
    assert profit == pytest.approx(profit, 1e-3)


@pytest.mark.parametrize(
    "W_0, S_0, W_11, W_10, S_11, S_10, N, R, H1",
    [
        (1.35, 44, 0.45, 4.72, 56, 24, 100, 1, -13.3),
        (1.08, 37, 0.48, 4.58, 59, 28, 100, 1, -13.2),
    ],
)
def test_quiz_2_10(W_0, S_0, W_11, W_10, S_11, S_10, N, R, H1) -> None:
    model = OneStepModel(
        asset_params=TerminalAssetParams([S_10, S_11], 1, S_0),
        derivative_params=TerminalDerivativeParams([W_10 * N, W_11 * N], 1),
        R=R,
    )
    H1 = model.H1
    assert pytest.approx(H1, 1) == H1


@pytest.mark.parametrize(
    "S, u, d, T, result",
    [(4, 2, 1 / 2, 3, [[4.0], [2.0, 8.0], [1.0, 4.0, 16.0], [1 / 2, 2, 8, 32]])],
)
def test_example_2_1(S, u, d, T, result) -> None:
    model = asset_factory(CRRAssetParams(S, u, d, T))
    for i in range(T + 1):
        np.testing.assert_almost_equal(model[i], result[i])


def test_example_2_2() -> None:
    K = 3
    u = 2
    d = 1 / 2
    S = 4
    R = 5 / 4
    steps = 3
    expire = 2
    asset = asset_factory(CRRAssetParams(S, u, d, steps))
    pi = pi_factory(CRRPiParams(R, S, u, d))
    derivative = derivative_factory(DerivativeParams(K, expire, type="call"))
    derivative.compute_grid(pi, asset)
    np.testing.assert_almost_equal(derivative[0], [2.40])
    np.testing.assert_almost_equal(derivative[1], [0.4, 5.6])
    np.testing.assert_almost_equal(derivative[2], [0, 1, 13])


def test_example_2_3() -> None:
    K = 3
    u = 2
    d = 1 / 2
    S = 4
    R = 5 / 4
    steps = 3
    expire = 3

    model = MultiStepModel(CRRAssetParams(S, u, d, steps), DerivativeParams(K, expire, type="call"), R)
    derivative = model.derivative
    np.testing.assert_almost_equal(derivative[0], [2.816])
    np.testing.assert_almost_equal(derivative[1], [0.8, 6.24])
    np.testing.assert_almost_equal(derivative[2], [0, 2, 13.60])
    np.testing.assert_almost_equal(derivative[3], [0, 0, 5, 29])


def test_example_2_4() -> None:
    K = 3
    u = 2
    d = 1 / 2
    S = 4
    R = 5 / 4
    steps = 3
    expire = 3
    model = MultiStepModel(CRRAssetParams(S, u, d, steps), DerivativeParams(K, expire, type="call"), R)
    derivative = model.derivative
    np.testing.assert_almost_equal(model.state_price[3], [8 / 125, 24 / 125, 24 / 125, 8 / 125])
    np.testing.assert_almost_equal(model.premium, derivative[0, 0])


def test_workshop_3_1() -> None:
    K = 23
    S = 20
    u = 1.06
    d = 0.95
    R = 1.01
    T = 10
    put_model = MultiStepModel(
        CRRAssetParams(S, u, d, T),
        DerivativeParams(K, T, type="put"),
        R=R,
    )
    put_premium = put_model.premium

    call_model = MultiStepModel(
        CRRAssetParams(S, u, d, T),
        DerivativeParams(K, T, type="call"),
        R=R,
    )
    call_premium = call_model.premium
    np.testing.assert_almost_equal(put_premium, 1.8202, 4)
    np.testing.assert_almost_equal(call_premium, 0.9986, 4)
    np.testing.assert_approx_equal(put_call_parity(S, K, R, T), call_premium - put_premium)


def test_workshop_3_2() -> None:
    K = 10
    S = 10
    u = 1.2
    d = 1 / u
    T = 8
    R = 1.05
    put_model = MultiStepModel(
        CRRAssetParams(S, u, d, T),
        DerivativeParams(K, T, type="put"),
        R=R,
    )
    np.testing.assert_almost_equal(
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
    np.testing.assert_almost_equal(put_model.premium, 0.4793, 4)
    np.testing.assert_almost_equal(put_model.derivative[0, 0], 0.4793, 4)


@pytest.mark.parametrize("S, u, d, T, largest, smallest", [(12, 1.2, 0.9, 3, 20.736, 8.748)])
def test_quiz_3_3(S, u, d, T, largest, smallest) -> None:
    asset = asset_factory(CRRAssetParams(S, u, d, T))
    np.testing.assert_almost_equal(asset[T, T], largest)
    np.testing.assert_almost_equal(asset[T, 0], smallest)


@pytest.mark.parametrize(
    "C, pi, R, T, premium",
    [([0, 0, 0, 3.84], 0.3, 1.13, 3, 0.0719), ([0, 0, 0, 3.5], 0.3, 1.18, 3, 0.0575)],
)
def test_quiz_3_4(C, pi, R, T, premium) -> None:
    model = MultiStepModel(
        TerminalAssetParams(C, T, 0),
        TerminalDerivativeParams(C, T),
        R=1.13,
        pi_values=TerminalPiParams(R=R, p_up=pi, p_down=1 - pi),
    )
    np.testing.assert_almost_equal(model.premium, premium, 4)


@pytest.mark.parametrize("K, T, asset, put", [(30, 3, [10, 15, 25, 35], [20, 15, 5, 0])])
def test_quiz_3_5(K, T, asset, put) -> None:
    asset = asset_factory(TerminalAssetParams(asset, T, 0))
    derivative = derivative_factory(DerivativeParams(K, T, type="put"))
    result = derivative.compute_terminal(asset)
    np.testing.assert_almost_equal(result, put)


@pytest.mark.parametrize("R, state_22, result", [(1.2, 0.27, 0.0984), (1.3, 0.12, 0.1788)])
def test_quiz_3_6(R, state_22, result):
    pi = np.sqrt(state_22 * R**2)
    pi_down = 1 - pi
    state_20 = (pi_down) ** 2 / R**2
    np.testing.assert_almost_equal(float(state_20), result, 4)


@pytest.mark.parametrize(
    "derivative, R, pi, premium",
    [([0, 0, 1, 0], 1.06, 0.5, 0.3149), ([0, 0, 1, 0], 1.05, 0.7, 0.3810)],
)
def test_quiz_3_7(derivative, R, pi, premium):
    T = len(derivative) - 1
    model = MultiStepModel(
        TerminalAssetParams([], T, 0),
        TerminalDerivativeParams(derivative, T),
        R=R,
        pi_values=TerminalPiParams(R=R, p_up=pi, p_down=1 - pi),
    )
    np.testing.assert_almost_equal(model.premium, premium, 4)


@pytest.mark.parametrize(
    "derivative, state_price, premium",
    [([0, 7, 13], [0, 0.1, 0.1], 2), ([0, 10, 14], [0, 0.3, 0.2], 5.8)],
)
def test_quiz_3_8(derivative, state_price, premium):
    result = np.array(derivative) @ np.array(state_price)
    np.testing.assert_almost_equal(float(result), premium, 4)


@pytest.mark.parametrize(
    "K, T, S, u, R, premium",
    [(8, 2, 12, 1.3, 1.07, 5.16), (9, 2, 11, 1.2, 1.06, 3.167)],
)
def test_quiz_3_9(K, T, S, u, R, premium):
    d = 1 / u
    model = MultiStepModel(CRRAssetParams(S, u, d, T), DerivativeParams(K, T, type="call"), R=R)
    np.testing.assert_almost_equal(model.premium, premium, 3)


@pytest.mark.parametrize(
    "K, T, S, u, R, premium",
    [(25, 2, 26, 1.4, 1.05, 2.773), (24, 2, 26, 1.4, 1.08, 2.004)],
)
def test_quiz_3_10(K, T, S, u, R, premium):
    d = 1 / u
    model = MultiStepModel(CRRAssetParams(S, u, d, T), DerivativeParams(K, T, type="put"), R=R)
    np.testing.assert_almost_equal(model.premium, premium, 3)
