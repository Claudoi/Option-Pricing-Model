# Pricing Montecarlo Placeholder
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

    Parameters:
        S (float): Initial stock price.
        K (float): Strike price of the option.
        T (float): Time to maturity in years.
        r (float): Risk-free interest rate (annualized).
        sigma (float): Volatility of the underlying asset (annualized).
        option_type (str): Type of option, either "call" or "put".
        n_simulations (int): Number of Monte Carlo simulations to run.
        n_steps (int): Number of time steps in each simulation.

    Returns:
        float: Monte Carlo estimated price of the Asian option.

    """


    # Validate input

    if option_type not in {"call", "put"}:
        raise ValueError("option_type must be 'call' or 'put'")
    if any(param <= 0 for param in [S, K, T, sigma, n_simulations, n_steps]):
        raise ValueError("All numeric inputs must be positive.")

    dt = T / n_steps
    drift = (r - 0.5 * sigma**2) * dt
    diffusion = sigma * np.sqrt(dt)
    discount = np.exp(-r * T)

    # Simulate log-returns and build paths

    random_shocks = np.random.normal(0, 1, size=(n_simulations, n_steps))
    log_returns = drift + diffusion * random_shocks
    log_paths = np.cumsum(log_returns, axis=1)
    price_paths = S * np.exp(log_paths)

    # Average price over the path (excluding S at t=0)

    avg_prices = price_paths.mean(axis=1)

    # Payoffs
    if option_type == "call":
        payoffs = np.maximum(avg_prices - K, 0)
    else:
        payoffs = np.maximum(K - avg_prices, 0)

    return discount * np.mean(payoffs)



import numpy as np


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

    if option_type == "call":
        intrinsic = ST > K
    else:
        intrinsic = ST < K

    final_payoff = payout * (barrier_crossed & intrinsic)
    return discount * np.mean(final_payoff)
