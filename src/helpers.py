def present_value(future_value: float, R: float, T: float) -> float:
    """Calculate the present value based on future value, interest, and accumulating period

    Args:
        future_value (float): future value
        R (float): interest rate
        T (float): accumulating period

    Returns:
        float: present value
    """
    return future_value / R**T


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


def calculate_up_state_prob(
    S_11: float,
    S_10: float,
    R: float,
    S_0: float,
) -> float:
    """Calculate pi for one step binomial model"""
    return (R * S_0 - S_10) / (S_11 - S_10)


def calculate_down_state_prob(
    S_11: float,
    S_10: float,
    R: float,
    S_0: float,
) -> float:
    """Calculate 1- pi for one step binomial model"""
    return (S_11 - R * S_0) / (S_11 - S_10)


def calculate_W_0_replicating(H0: float, H1: float, S_0: float) -> float:
    """Calculate option premium using the replicating portfolio method"""
    return H0 + H1 * S_0


def calculate_W_0_general(
    W_11: float, W_10: float, p_up: float, p_down: float, R: float
) -> float:
    """Calculate the option premium using the general pricing formula"""
    return 1 / R * (p_up * W_11 + p_down * W_10)
