import numpy as np
from scipy.optimize import minimize


def sabr_volatility(F, K, T, alpha, beta, rho, nu):
    """
    Hagan's SABR approximation formula for implied volatility.
    """
    if F == K:
        numer = alpha
        denom = F**(1 - beta)
        term1 = ((1 - beta)**2 / 24) * (alpha**2) / (F**(2 - 2 * beta))
        term2 = (rho * beta * nu * alpha) / (4 * F**(1 - beta))
        term3 = ((2 - 3 * rho**2) * nu**2) / 24
        return (alpha / denom) * (1 + (term1 + term2 + term3) * T)
    else:
        logFK = np.log(F / K)
        z = (nu / alpha) * (F * K)**((1 - beta) / 2) * logFK
        x_z = np.log((np.sqrt(1 - 2 * rho * z + z**2) + z - rho) / (1 - rho))
        numer = alpha * z
        denom = (F * K)**((1 - beta) / 2) * (1 + ((1 - beta)**2 / 24) * logFK**2 + ((1 - beta)**4 / 1920) * logFK**4)
        term1 = 1 + (((1 - beta)**2 / 24) * (alpha**2) / ((F * K)**(1 - beta))
                     + (rho * beta * nu * alpha) / (4 * (F * K)**((1 - beta) / 2))
                     + ((2 - 3 * rho**2) * nu**2) / 24) * T
        return (numer / denom) * (z / x_z) * term1


def objective_sabr(params, F, K, T, market_vols):
    alpha, beta, rho, nu = params
    model_vols = [sabr_volatility(F, k, T, alpha, beta, rho, nu) for k in K]
    return np.mean((np.array(model_vols) - market_vols) ** 2)


def calibrate_sabr(F, K, T, market_vols, beta_fixed=0.5):
    """
    Calibrate SABR parameters using market data.
    """
    x0 = [0.2, beta_fixed, 0.0, 0.5]  # alpha, beta, rho, nu
    bounds = [
        (0.001, 2.0),   # alpha
        (beta_fixed, beta_fixed),  # beta fixed
        (-0.999, 0.999),  # rho
        (0.001, 2.0)    # nu
    ]

    result = minimize(objective_sabr, x0, args=(F, K, T, market_vols), bounds=bounds)

    if not result.success:
        raise ValueError("SABR calibration failed.")

    return result.x
