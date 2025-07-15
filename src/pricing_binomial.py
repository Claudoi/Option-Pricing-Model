import numpy as np
from src.constants import OPTION_TYPES



def _validate_inputs(S: float, K: float, T: float, r: float, sigma: float, N: int, option_type: str):
    if S <= 0 or K <= 0 or T <= 0 or sigma <= 0 or N <= 0:
        raise ValueError("All input parameters must be positive and non-zero.")
    if option_type not in OPTION_TYPES:
        raise ValueError("option_type must be either 'call' or 'put'.")



def _binomial_parameters(T: float, r: float, sigma: float, N: int, q: float = 0.0):
    dt = T / N
    u = np.exp(sigma * np.sqrt(dt))
    d = 1 / u
    discount = np.exp(-r * dt)
    p = (np.exp((r - q) * dt) - d) / (u - d)
    return dt, u, d, p, discount



def binomial_european(
    S: float,
    K: float,
    T: float,
    r: float,
    sigma: float,
    N: int,
    option_type: str = "call",
    q: float = 0.0,
    return_tree: bool = False
) -> float | tuple[float, np.ndarray]:
    
    """
    Compute the price of a European option using the binomial model.
    """
    
    _validate_inputs(S, K, T, r, sigma, N, option_type)
    dt, u, d, p, discount = _binomial_parameters(T, r, sigma, N, q)

    ST = np.array([S * (u ** (N - j)) * (d ** j) for j in range(N + 1)])

    if option_type == "call":
        option_values = np.maximum(ST - K, 0)
    else:
        option_values = np.maximum(K - ST, 0)

    for i in range(N - 1, -1, -1):
        option_values = discount * (p * option_values[:-1] + (1 - p) * option_values[1:])

    if return_tree:
        return option_values[0], ST
    return option_values[0]



def binomial_american(
    S: float,
    K: float,
    T: float,
    r: float,
    sigma: float,
    N: int,
    option_type: str = "call",
    q: float = 0.0,
    return_tree: bool = False
) -> float | tuple[float, np.ndarray]:
    
    """
    Compute the price of an American option using the binomial model.
    """

    _validate_inputs(S, K, T, r, sigma, N, option_type)
    dt, u, d, p, discount = _binomial_parameters(T, r, sigma, N, q)

    asset_tree = np.zeros((N + 1, N + 1))
    for i in range(N + 1):
        for j in range(i + 1):
            asset_tree[j, i] = S * (u ** (i - j)) * (d ** j)

    option_tree = np.zeros_like(asset_tree)
    for j in range(N + 1):
        if option_type == "call":
            option_tree[j, N] = max(0, asset_tree[j, N] - K)
        else:
            option_tree[j, N] = max(0, K - asset_tree[j, N])

    for i in range(N - 1, -1, -1):
        for j in range(i + 1):
            continuation = discount * (p * option_tree[j, i + 1] + (1 - p) * option_tree[j + 1, i + 1])
            if option_type == "call":
                exercise = max(0, asset_tree[j, i] - K)
            else:
                exercise = max(0, K - asset_tree[j, i])
            option_tree[j, i] = max(continuation, exercise)

    if return_tree:
        return option_tree[0, 0], asset_tree
    return option_tree[0, 0]



def binomial_price_and_tree(
    S: float,
    K: float,
    T: float,
    r: float,
    sigma: float,
    N: int,
    option_type: str = "call",
    american: bool = False,
    q: float = 0.0
) -> tuple[float, np.ndarray]:
    
    """
    Wrapper to compute either European or American option price and tree.
    """

    if american:
        return binomial_american(S, K, T, r, sigma, N, option_type, q, return_tree=True)
    else:
        return binomial_european(S, K, T, r, sigma, N, option_type, q, return_tree=True)
