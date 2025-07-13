import numpy as np
from scipy.stats import norm

def _d1(S, K, T, r, sigma):
    return (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))


def _d2(S, K, T, r, sigma):
    return _d1(S, K, T, r, sigma) - sigma * np.sqrt(T)


def delta(S, K, T, r, sigma, option_type='call'):
    d1 = _d1(S, K, T, r, sigma)
    if option_type == 'call':
        return norm.cdf(d1)
    elif option_type == 'put':
        return norm.cdf(d1) - 1
    raise ValueError("option_type must be 'call' or 'put'")


def gamma(S, K, T, r, sigma):
    d1 = _d1(S, K, T, r, sigma)
    return norm.pdf(d1) / (S * sigma * np.sqrt(T))


def vega(S, K, T, r, sigma):
    d1 = _d1(S, K, T, r, sigma)
    return (S * norm.pdf(d1) * np.sqrt(T)) / 100  # per 1% change in vol


def theta(S, K, T, r, sigma, option_type='call'):
    d1 = _d1(S, K, T, r, sigma)
    d2 = _d2(S, K, T, r, sigma)
    term1 = -(S * norm.pdf(d1) * sigma) / (2 * np.sqrt(T))

    if option_type == 'call':
        term2 = r * K * np.exp(-r * T) * norm.cdf(d2)
        return (term1 - term2) / 365  # per day

    elif option_type == 'put':
        term2 = r * K * np.exp(-r * T) * norm.cdf(-d2)
        return (term1 + term2) / 365  # per day

    raise ValueError("option_type must be 'call' or 'put'")


def rho(S, K, T, r, sigma, option_type='call'):
    d2 = _d2(S, K, T, r, sigma)

    if option_type == 'call':
        return (K * T * np.exp(-r * T) * norm.cdf(d2)) / 100  # per 1% change in rate

    elif option_type == 'put':
        return (-K * T * np.exp(-r * T) * norm.cdf(-d2)) / 100
        
    raise ValueError("option_type must be 'call' or 'put'")