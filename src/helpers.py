def present_value(future_value: float, R: float, T: float) -> float:
    return future_value / R**T


def put_call_parity(asset: float, strike: float, R: float, T: float = 0) -> float:
    return asset - present_value(strike, R, T)


def calculate_H1(
    S_11: float,
    S_10: float,
    W_11: float,
    W_10: float,
) -> float:
    return (W_11 - W_10) / (S_11 - S_10)


def calculate_H0(
    S_11: float,
    S_10: float,
    W_11: float,
    W_10: float,
    R: float,
) -> float:
    return (S_11 * W_10 - S_10 * W_11) / (S_11 - S_10) / R


def calculate_up_state_prob(
    S_11: float,
    S_10: float,
    R: float,
    S_0: float,
) -> float:
    return (R * S_0 - S_10) / (S_11 - S_10)


def calculate_down_state_prob(
    S_11: float,
    S_10: float,
    R: float,
    S_0: float,
) -> float:
    return (S_11 - R * S_0) / (S_11 - S_10)


def calculate_W_0_replicating(H0: float, H1: float, S_0: float) -> float:
    return H0 + H1 * S_0


def calculate_W_0_general(
    W_11: float, W_10: float, p_up: float, p_down: float, R: float
) -> float:
    return 1 / R * (p_up * W_11 + p_down * W_10)
