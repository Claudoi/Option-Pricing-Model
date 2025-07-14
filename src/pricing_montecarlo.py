import numpy as np


def monte_carlo_asian(
    S: float,
    K: float,
    T: float,
    r: float,
    sigma: float,
    option_type: str = "call",
    n_simulations: int = 10_000,
    n_steps: int = 100
) -> float:
    
    """
    Price an arithmetic-average Asian option using Monte Carlo simulation.
    """

    if option_type not in {"call", "put"}:
        raise ValueError("option_type must be 'call' or 'put'")
    if any(param <= 0 for param in [S, K, T, sigma, n_simulations, n_steps]):
        raise ValueError("All numeric inputs must be positive.")

    dt = T / n_steps
    drift = (r - 0.5 * sigma**2) * dt
    diffusion = sigma * np.sqrt(dt)
    discount = np.exp(-r * T)

    random_shocks = np.random.normal(0, 1, size=(n_simulations, n_steps))
    log_returns = drift + diffusion * random_shocks
    log_paths = np.cumsum(log_returns, axis=1)
    price_paths = S * np.exp(log_paths)
    avg_prices = price_paths.mean(axis=1)

    if option_type == "call":
        payoffs = np.maximum(avg_prices - K, 0)
    else:
        payoffs = np.maximum(K - avg_prices, 0)

    return discount * np.mean(payoffs)



def monte_carlo_asian_geometric(
    S: float,
    K: float,
    T: float,
    r: float,
    sigma: float,
    option_type: str = "call",
    n_simulations: int = 10000,
    n_steps: int = 100
) -> float:
    """
    Price a geometric-average Asian option using Monte Carlo simulation.
    """
    if option_type not in {"call", "put"}:
        raise ValueError("option_type must be 'call' or 'put'")

    dt = T / n_steps
    drift = (r - 0.5 * sigma**2) * dt
    diffusion = sigma * np.sqrt(dt)
    discount = np.exp(-r * T)

    Z = np.random.normal(0, 1, size=(n_simulations, n_steps))
    log_returns = drift + diffusion * Z
    log_paths = np.cumsum(log_returns, axis=1)
    paths = S * np.exp(log_paths)

    geo_avg = np.exp(np.mean(np.log(paths), axis=1))
    if option_type == "call":
        payoffs = np.maximum(geo_avg - K, 0)
    else:
        payoffs = np.maximum(K - geo_avg, 0)

    return discount * np.mean(payoffs)




def monte_carlo_digital_barrier(
    S: float,
    K: float,
    T: float,
    r: float,
    sigma: float,
    barrier: float,
    option_type: str = "call",
    barrier_type: str = "up-and-in",
    payout: float = 1.0,
    n_simulations: int = 10000,
    n_steps: int = 100
) -> float:
    
    if option_type not in {"call", "put"}:
        raise ValueError("option_type must be 'call' or 'put'")
    if barrier_type not in {"up-and-in", "up-and-out", "down-and-in", "down-and-out"}:
        raise ValueError("barrier_type must be one of 'up-and-in', 'up-and-out', 'down-and-in', 'down-and-out'")

    dt = T / n_steps
    discount = np.exp(-r * T)
    Z = np.random.normal(0, 1, (n_simulations, n_steps))
    paths = np.zeros_like(Z)
    paths[:, 0] = S

    for t in range(1, n_steps):
        paths[:, t] = paths[:, t-1] * np.exp((r - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * Z[:, t])

    barrier_crossed = {
        "up-and-in":  np.any(paths >= barrier, axis=1),
        "up-and-out": np.all(paths <  barrier, axis=1),
        "down-and-in":  np.any(paths <= barrier, axis=1),
        "down-and-out": np.all(paths >  barrier, axis=1),
    }[barrier_type]

    ST = paths[:, -1]
    intrinsic = ST > K if option_type == "call" else ST < K

    final_payoff = payout * (barrier_crossed & intrinsic)
    return discount * np.mean(final_payoff)


def monte_carlo_lookback(
    S: float,
    K: float,
    T: float,
    r: float,
    sigma: float,
    option_type: str = "call",
    strike_type: str = "fixed",
    n_simulations: int = 10000,
    n_steps: int = 100
) -> float:
    
    """
    Prices European-style lookback options using Monte Carlo simulation.
    """

    if option_type not in {"call", "put"}:
        raise ValueError("option_type must be either 'call' or 'put'")
    if strike_type not in {"fixed", "floating"}:
        raise ValueError("strike_type must be either 'fixed' or 'floating'")
    if any(param <= 0 for param in [S, K, T, sigma, n_simulations, n_steps]):
        raise ValueError("S, K, T, sigma, n_simulations and n_steps must be positive.")

    dt = T / n_steps
    discount = np.exp(-r * T)
    Z = np.random.normal(0, 1, size=(n_simulations, n_steps))
    paths = np.zeros_like(Z)
    paths[:, 0] = S

    drift = (r - 0.5 * sigma**2) * dt
    diffusion = sigma * np.sqrt(dt)
    for t in range(1, n_steps):
        paths[:, t] = paths[:, t - 1] * np.exp(drift + diffusion * Z[:, t])

    S_T = paths[:, -1]
    S_max = np.max(paths, axis=1)
    S_min = np.min(paths, axis=1)

    if strike_type == "fixed":
        payoffs = np.maximum(S_max - K, 0) if option_type == "call" else np.maximum(K - S_min, 0)
    else:
        payoffs = np.maximum(S_T - S_min, 0) if option_type == "call" else np.maximum(S_max - S_T, 0)

    return discount * np.mean(payoffs)


def monte_carlo_vanilla(
    S: float,
    K: float,
    T: float,
    r: float,
    sigma: float,
    option_type: str = "call",
    n_simulations: int = 10000,
    n_steps: int = 100
) -> float:
    
    """
    Price a European vanilla option using Monte Carlo simulation.
    """

    if option_type not in {"call", "put"}:
        raise ValueError("option_type must be either 'call' or 'put'")
    if any(param <= 0 for param in [S, K, T, sigma, n_simulations, n_steps]):
        raise ValueError("All numeric inputs must be positive.")

    dt = T / n_steps
    drift = (r - 0.5 * sigma**2) * dt
    diffusion = sigma * np.sqrt(dt)
    discount = np.exp(-r * T)

    Z = np.random.normal(0, 1, (n_simulations, n_steps))
    paths = np.zeros_like(Z)
    paths[:, 0] = S

    for t in range(1, n_steps):
        paths[:, t] = paths[:, t-1] * np.exp(drift + diffusion * Z[:, t])

    ST = paths[:, -1]
    payoffs = np.maximum(ST - K, 0) if option_type == "call" else np.maximum(K - ST, 0)

    return discount * np.mean(payoffs)




def monte_carlo_barrier_knock_in_out_payoff_paths(
    S: float,
    K: float,
    T: float,
    r: float,
    sigma: float,
    barrier: float,
    option_type: str = "call",
    barrier_type: str = "up-and-in",
    payout: float = 1.0,
    n_simulations: int = 10000,
    n_steps: int = 100
) -> float:
    
    """
    Monte Carlo pricing for knock-in/out barrier digital options with path-dependent checks.
    """

    if option_type not in {"call", "put"}:
        raise ValueError("option_type must be 'call' or 'put'")
    if barrier_type not in {"up-and-in", "up-and-out", "down-and-in", "down-and-out"}:
        raise ValueError("Invalid barrier type")

    dt = T / n_steps
    discount = np.exp(-r * T)

    Z = np.random.normal(0, 1, size=(n_simulations, n_steps))
    paths = np.zeros_like(Z)
    paths[:, 0] = S
    for t in range(1, n_steps):
        paths[:, t] = paths[:, t-1] * np.exp((r - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * Z[:, t])

    ST = paths[:, -1]

    if option_type == "call":
        payoffs = ST > K
    else:
        payoffs = ST < K

    hit_barrier = {
        "up-and-in": np.any(paths >= barrier, axis=1),
        "up-and-out": np.all(paths < barrier, axis=1),
        "down-and-in": np.any(paths <= barrier, axis=1),
        "down-and-out": np.all(paths > barrier, axis=1),
    }[barrier_type]

    final_payoff = payout * (payoffs & hit_barrier)
    return discount * np.mean(final_payoff)




def monte_carlo_greeks(
    greek: str,
    S: float,
    K: float,
    T: float,
    r: float,
    sigma: float,
    option_type: str = "call",
    h: float = 1e-2,
    n_simulations: int = 10000,
    n_steps: int = 100
) -> float:
    
    """
    Estimate Greek sensitivities using finite differences on vanilla Monte Carlo.
    """
    
    if greek == "delta":
        price_up = monte_carlo_vanilla(S + h, K, T, r, sigma, option_type, n_simulations, n_steps)
        price_down = monte_carlo_vanilla(S - h, K, T, r, sigma, option_type, n_simulations, n_steps)
        return (price_up - price_down) / (2 * h)
    elif greek == "vega":
        price_up = monte_carlo_vanilla(S, K, T, r, sigma + h, option_type, n_simulations, n_steps)
        price_down = monte_carlo_vanilla(S, K, T, r, sigma - h, option_type, n_simulations, n_steps)
        return (price_up - price_down) / (2 * h)
    elif greek == "rho":
        price_up = monte_carlo_vanilla(S, K, T, r + h, sigma, option_type, n_simulations, n_steps)
        price_down = monte_carlo_vanilla(S, K, T, r - h, sigma, option_type, n_simulations, n_steps)
        return (price_up - price_down) / (2 * h)
    elif greek == "theta":
        price_up = monte_carlo_vanilla(S, K, T + h, r, sigma, option_type, n_simulations, n_steps)
        price_down = monte_carlo_vanilla(S, K, T - h, r, sigma, option_type, n_simulations, n_steps)
        return (price_up - price_down) / (2 * h)
    else:
        raise ValueError("Unsupported greek. Use 'delta', 'vega', 'rho', or 'theta'.")

