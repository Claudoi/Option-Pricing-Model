import numpy as np

def binomial_american(
    S: float,
    K: float,
    T: float,
    r: float,
    sigma: float,
    N: int,
    option_type: str = "call"
) -> float:
 

    if S <= 0 or K <= 0 or T <= 0 or sigma <= 0 or N <= 0:
        raise ValueError("All input parameters must be positive and non-zero.")

    if option_type not in {"call", "put"}:
        raise ValueError("option_type must be either 'call' or 'put'.")

    dt = T / N
    discount = np.exp(-r * dt)
    u = np.exp(sigma * np.sqrt(dt))
    d = 1 / u
    p = (np.exp(r * dt) - d) / (u - d)

    # Build asset price tree
    asset_tree = np.zeros((N + 1, N + 1))
    for i in range(N + 1):
        for j in range(i + 1):
            asset_tree[j, i] = S * (u ** (i - j)) * (d ** j)

    # Initialize option values at maturity
    option_tree = np.zeros((N + 1, N + 1))
    for j in range(N + 1):
        if option_type == "call":
            option_tree[j, N] = max(0, asset_tree[j, N] - K)
        else:
            option_tree[j, N] = max(0, K - asset_tree[j, N])

    # Backward induction
    for i in range(N - 1, -1, -1):
        for j in range(i + 1):
            continuation = discount * (p * option_tree[j, i + 1] + (1 - p) * option_tree[j + 1, i + 1])
            if option_type == "call":
                exercise = max(0, asset_tree[j, i] - K)
            else:
                exercise = max(0, K - asset_tree[j, i])
            option_tree[j, i] = max(continuation, exercise)

    return option_tree[0, 0]
