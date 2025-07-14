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
    """

    if S <= 0 or K <= 0 or T <= 0 or sigma <= 0:
        raise ValueError("S, K, T and sigma must be positive and non-zero.")

    option_type = option_type.lower()
    if option_type not in {"call", "put"}:
        raise ValueError("option_type must be either 'call' or 'put'.")

    sqrt_T = np.sqrt(T)
    discounted_S = S * np.exp(-q * T)

    d1 = (np.log(S / K) + (r - q + 0.5 * sigma ** 2) * T) / (sigma * sqrt_T)
    d2 = d1 - sigma * sqrt_T

    payoff = {
        "call": lambda: discounted_S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2),
        "put":  lambda: K * np.exp(-r * T) * norm.cdf(-d2) - discounted_S * norm.cdf(-d1)
    }

    return payoff[option_type]()




def black_scholes_price_and_greeks(
    S: float,
    K: float,
    T: float,
    r: float,
    sigma: float,
    option_type: str = "call",
    q: float = 0.0
) -> dict:
    
    """
    Compute the Black-Scholes price and Greeks for a European option.
    """

    sqrt_T = np.sqrt(T)
    d1 = (np.log(S / K) + (r - q + 0.5 * sigma**2) * T) / (sigma * sqrt_T)
    d2 = d1 - sigma * sqrt_T

    price = black_scholes(S, K, T, r, sigma, option_type, q)

    delta = (
        np.exp(-q * T) * norm.cdf(d1) if option_type == "call"
        else np.exp(-q * T) * (norm.cdf(d1) - 1)
    )
    gamma = np.exp(-q * T) * norm.pdf(d1) / (S * sigma * sqrt_T)
    vega = S * np.exp(-q * T) * norm.pdf(d1) * sqrt_T / 100
    theta = (
        (-S * norm.pdf(d1) * sigma * np.exp(-q * T) / (2 * sqrt_T) - r * K * np.exp(-r * T) * norm.cdf(d2)) / 365
        if option_type == "call"
        else (-S * norm.pdf(d1) * sigma * np.exp(-q * T) / (2 * sqrt_T) + r * K * np.exp(-r * T) * norm.cdf(-d2)) / 365
    )
    rho = (
        K * T * np.exp(-r * T) * norm.cdf(d2) / 100
        if option_type == "call"
        else -K * T * np.exp(-r * T) * norm.cdf(-d2) / 100
    )

    return {
        "price": price,
        "delta": delta,
        "gamma": gamma,
        "vega": vega,
        "theta": theta,
        "rho": rho
    }




def implied_volatility_newton(
    market_price: float,
    S: float,
    K: float,
    T: float,
    r: float,
    option_type: str = "call",
    q: float = 0.0,
    tol: float = 1e-6,
    max_iter: int = 100
) -> float:
    
    """
    Estimate implied volatility using Newton-Raphson method.
    """

    sigma = 0.2  # initial guess
    for i in range(max_iter):
        price = black_scholes(S, K, T, r, sigma, option_type, q)
        sqrt_T = np.sqrt(T)
        d1 = (np.log(S / K) + (r - q + 0.5 * sigma**2) * T) / (sigma * sqrt_T)
        vega = S * np.exp(-q * T) * norm.pdf(d1) * sqrt_T

        price_diff = price - market_price
        if abs(price_diff) < tol:
            return sigma

        sigma -= price_diff / vega

    raise RuntimeError("Implied volatility did not converge")
