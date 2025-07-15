import numpy as np
from scipy.stats import norm
from src.constants import (OPTION_TYPES, ERROR_OPTION_TYPE, VEGA_SCALE, RHO_SCALE, THETA_SCALE)

def _d1(S, K, T, r, sigma):
    return (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))

def _d2(S, K, T, r, sigma):
    return _d1(S, K, T, r, sigma) - sigma * np.sqrt(T)

def delta(S, K, T, r, sigma, option_type='call'):
    if option_type not in OPTION_TYPES:
        raise ValueError(ERROR_OPTION_TYPE)
    d1 = _d1(S, K, T, r, sigma)
    return norm.cdf(d1) if option_type == 'call' else norm.cdf(d1) - 1

def gamma(S, K, T, r, sigma):
    d1 = _d1(S, K, T, r, sigma)
    return norm.pdf(d1) / (S * sigma * np.sqrt(T))

def vega(S, K, T, r, sigma):
    d1 = _d1(S, K, T, r, sigma)
    return (S * norm.pdf(d1) * np.sqrt(T)) * VEGA_SCALE

def theta(S, K, T, r, sigma, option_type='call'):
    if option_type not in OPTION_TYPES:
        raise ValueError(ERROR_OPTION_TYPE)

    d1 = _d1(S, K, T, r, sigma)
    d2 = _d2(S, K, T, r, sigma)
    term1 = -(S * norm.pdf(d1) * sigma) / (2 * np.sqrt(T))

    if option_type == 'call':
        term2 = r * K * np.exp(-r * T) * norm.cdf(d2)
        return (term1 - term2) * THETA_SCALE
    else:
        term2 = r * K * np.exp(-r * T) * norm.cdf(-d2)
        return (term1 + term2) * THETA_SCALE

def rho(S, K, T, r, sigma, option_type='call'):
    if option_type not in OPTION_TYPES:
        raise ValueError(ERROR_OPTION_TYPE)

    d2 = _d2(S, K, T, r, sigma)
    factor = K * T * np.exp(-r * T)
    return factor * norm.cdf(d2) * RHO_SCALE if option_type == 'call' else -factor * norm.cdf(-d2) * RHO_SCALE
