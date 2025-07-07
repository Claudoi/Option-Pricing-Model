import numpy as np
from scipy.stats import norm


def black_scholes(
    S: float,
    K: float,
    T: float,
    r: float,
    sigma: float,
    option_type: str = "call",
    q: float = 0.0
) -> float:


    """
    Compute the Black-Scholes price for a European option.

    Parameters
    ----------
    S : float
        Current price of the underlying asset
    K : float
        Strike price
    T : float
        Time to maturity in years
    r : float
        Annual risk-free interest rate (as decimal)
    sigma : float
        Annual volatility of the underlying asset (as decimal)
    option_type : str, optional
        Type of the option: 'call' or 'put' (default is 'call')
    q : float, optional
        Continuous dividend yield (default is 0.0)

    Returns
    -------
    float
        Option price according to Black-Scholes model
    """


    # --- Input validation ---

    if S <= 0 or K <= 0 or T <= 0 or sigma <= 0:
        raise ValueError("S, K, T and sigma must be positive and non-zero.")

    option_type = option_type.lower()
    if option_type not in {"call", "put"}:
        raise ValueError("option_type must be either 'call' or 'put'.")


    # --- Calculations ---

    sqrt_T = np.sqrt(T)
    discounted_S = S * np.exp(-q * T)

    d1 = (np.log(S / K) + (r - q + 0.5 * sigma ** 2) * T) / (sigma * sqrt_T)
    d2 = d1 - sigma * sqrt_T


    # --- Payoff logic using dispatch dictionary ---

    payoff = {
        "call": lambda: discounted_S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2),
        "put":  lambda: K * np.exp(-r * T) * norm.cdf(-d2) - discounted_S * norm.cdf(-d1)
    }

    return payoff[option_type]()
