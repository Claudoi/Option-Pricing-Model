import numpy as np
from scipy.optimize import minimize


def svi_total_variance(k, a, b, rho, m, sigma):
    return a + b * (rho * (k - m) + np.sqrt((k - m) ** 2 + sigma ** 2))


def objective_svi(params, k, market_variance):
    a, b, rho, m, sigma = params
    model = svi_total_variance(k, a, b, rho, m, sigma)
    return np.mean((model - market_variance) ** 2)


def calibrate_svi(log_moneyness, implied_vol, T):
    w_market = (implied_vol ** 2) * T

    x0 = [0.1, 0.1, 0.0, 0.0, 0.1]
    bounds = [
        (-1, 1),
        (0.001, 10),
        (-0.999, 0.999),
        (-1, 1),
        (0.001, 2)
    ]

    result = minimize(objective_svi, x0, args=(log_moneyness, w_market), bounds=bounds)

    if not result.success:
        raise ValueError("SVI calibration failed.")

    params = result.x
    w_fit = svi_total_variance(log_moneyness, *params)
    implied_vol_fit = np.sqrt(w_fit / T)
    
    return params, implied_vol_fit
