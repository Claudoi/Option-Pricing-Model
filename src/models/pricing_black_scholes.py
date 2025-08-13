import numpy as np
from scipy.stats import norm
from src.utils.constants import ONE_OVER_100, ONE_OVER_365, EPS_T, EPS_SIG
from src.utils.utils import validate_option_type, validate_positive_inputs


class BlackScholesOption:
    """
    Black-Scholes option pricing model.
    Computes price and Greeks for European calls and puts.
    """

    def __init__(self, S, K, T, r, sigma, option_type="call", q=0.0):
        self.S = S
        self.K = K
        self.T = T
        self.r = r
        self.sigma = sigma
        self.option_type = option_type.lower()
        self.q = q
        self._validate_inputs()

    def _validate_inputs(self):
        validate_option_type(self.option_type)
        validate_positive_inputs(self.S, self.K, self.T, self.sigma)

    def _d1(self):
        return (np.log(self.S / self.K) + (self.r - self.q + 0.5 * self.sigma**2) * self.T) / (self.sigma * np.sqrt(self.T))

    def _d2(self):
        return self._d1() - self.sigma * np.sqrt(self.T)

    def price(self):
        """
        Closed-form Black-Scholes price for call/put with continuous dividend yield q.
        """
        d1 = self._d1()
        d2 = self._d2()
        disc_r = np.exp(-self.r * self.T)
        disc_q = np.exp(-self.q * self.T)
        if self.option_type == "call":
            return self.S * disc_q * norm.cdf(d1) - self.K * disc_r * norm.cdf(d2)
        else:
            return self.K * disc_r * norm.cdf(-d2) - self.S * disc_q * norm.cdf(-d1)

    def greeks(self):
        """
        Greeks with your conventions:
        - vega scaled per 1% vol change (ONE_OVER_100)
        - theta per day (ONE_OVER_365)
        - rho per 1% rate change (ONE_OVER_100)
        """
        d1 = self._d1()
        d2 = self._d2()
        sqrt_T = np.sqrt(self.T)
        disc_r = np.exp(-self.r * self.T)
        disc_q = np.exp(-self.q * self.T)

        delta = disc_q * norm.cdf(d1) if self.option_type == "call" else disc_q * (norm.cdf(d1) - 1.0)
        gamma = disc_q * norm.pdf(d1) / (self.S * self.sigma * sqrt_T)
        vega  = self.S * disc_q * norm.pdf(d1) * sqrt_T * ONE_OVER_100

        # Note: this theta matches your original convention (no explicit q term).
        theta = (
            (-(self.S * norm.pdf(d1) * self.sigma * disc_q) / (2 * sqrt_T) - self.r * self.K * disc_r * norm.cdf(d2)) * ONE_OVER_365
            if self.option_type == "call"
            else (-(self.S * norm.pdf(d1) * self.sigma * disc_q) / (2 * sqrt_T) + self.r * self.K * disc_r * norm.cdf(-d2)) * ONE_OVER_365
        )

        rho = (
            self.K * self.T * disc_r * norm.cdf(d2) * ONE_OVER_100
            if self.option_type == "call"
            else -self.K * self.T * disc_r * norm.cdf(-d2) * ONE_OVER_100
        )

        return {"price": self.price(), "delta": delta, "gamma": gamma, "vega": vega, "theta": theta, "rho": rho}


# ---------- Vectorized helpers (for heatmaps, surfaces, etc.) ----------

def _safe_T_sigma(T, sigma):
    """
    Clip T and sigma to small positive thresholds (EPS_T, EPS_SIG) to avoid divide-by-zero and NaNs.
    """
    T = np.asarray(T, dtype=float)
    sigma = np.asarray(sigma, dtype=float)
    T_safe = np.maximum(T, EPS_T)
    sigma_safe = np.maximum(sigma, EPS_SIG)
    return T_safe, sigma_safe

def _d1_d2_vec(S, K, T, r, q, sigma):
    """
    Vectorized d1, d2. S and sigma can be arrays (e.g., meshgrids).
    K, T, r, q can be scalars or broadcastable arrays.
    Returns d1, d2 and the clipped T/sigma (T_safe, sigma_safe).
    """
    S = np.asarray(S, dtype=float)
    K = np.asarray(K, dtype=float)
    r = np.asarray(r, dtype=float)
    q = np.asarray(q, dtype=float)
    T_safe, sigma_safe = _safe_T_sigma(T, sigma)

    sqrt_T = np.sqrt(T_safe)
    with np.errstate(divide='ignore', invalid='ignore'):
        d1 = (np.log(S / K) + (r - q + 0.5 * sigma_safe**2) * T_safe) / (sigma_safe * sqrt_T)
        d2 = d1 - sigma_safe * sqrt_T
    return d1, d2, T_safe, sigma_safe

def bs_price_vectorized(S, K, T, r, sigma, option_type="call", q=0.0):
    """
    Vectorized Black-Scholes price with stability handling.
    Falls back to discounted intrinsic value when T or sigma are ~0.
    """
    d1, d2, T_safe, sigma_safe = _d1_d2_vec(S, K, T, r, q, sigma)
    disc_r = np.exp(-r * T_safe)
    disc_q = np.exp(-q * T_safe)

    if option_type == "call":
        price = S * disc_q * norm.cdf(d1) - K * disc_r * norm.cdf(d2)
    else:
        price = K * disc_r * norm.cdf(-d2) - S * disc_q * norm.cdf(-d1)

    # Use T_safe/sigma_safe for the near-maturity check
    near_maturity = (T_safe <= EPS_T) | (sigma_safe <= EPS_SIG)
    if np.any(near_maturity):
        intrinsic = S * disc_q - K * disc_r
        intrinsic = np.where(option_type == "call", np.maximum(intrinsic, 0.0), np.maximum(-intrinsic, 0.0))
        price = np.where(near_maturity, intrinsic, price)

    return price

def bs_greeks_vectorized(S, K, T, r, sigma, option_type="call", q=0.0, theta_includes_q=False):
    """
    Vectorized Greeks with your scaling conventions.
    Optional `theta_includes_q=True` adds the dividend yield term to theta.
    """
    d1, d2, T_safe, sigma_safe = _d1_d2_vec(S, K, T, r, q, sigma)
    sqrt_T = np.sqrt(T_safe)
    disc_r = np.exp(-r * T_safe)
    disc_q = np.exp(-q * T_safe)

    Nd1 = norm.cdf(d1)
    Nd2 = norm.cdf(d2)
    Nmd2 = norm.cdf(-d2)
    nd1 = norm.pdf(d1)

    if option_type == "call":
        delta = disc_q * Nd1
        theta = (-(S * nd1 * sigma_safe * disc_q) / (2 * sqrt_T) - r * K * disc_r * Nd2)
        if theta_includes_q:
            theta += q * S * disc_q * Nd1
        rho = (K * T_safe * disc_r * Nd2) * ONE_OVER_100
    else:
        delta = disc_q * (Nd1 - 1.0)
        theta = (-(S * nd1 * sigma_safe * disc_q) / (2 * sqrt_T) + r * K * disc_r * Nmd2)
        if theta_includes_q:
            theta -= q * S * disc_q * norm.cdf(-d1)
        rho = (-K * T_safe * disc_r * Nmd2) * ONE_OVER_100

    gamma = disc_q * nd1 / (S * sigma_safe * sqrt_T)
    vega  = S * disc_q * nd1 * sqrt_T * ONE_OVER_100

    # Theta per day
    theta *= ONE_OVER_365

    # Use T_safe/sigma_safe for the near-maturity check (consistent with price)
    near_maturity = (T_safe <= EPS_T) | (sigma_safe <= EPS_SIG)
    if np.any(near_maturity):
        # Stable limiting values; customize if you prefer different limits
        delta_lim = np.where(option_type == "call", (S > K).astype(float), -(S < K).astype(float))
        delta = np.where(near_maturity, delta_lim, delta)
        gamma = np.where(near_maturity, 0.0, gamma)
        vega  = np.where(near_maturity, 0.0, vega)
        theta = np.where(near_maturity, 0.0, theta)
        rho   = np.where(near_maturity, 0.0, rho)

    return {"delta": delta, "gamma": gamma, "vega": vega, "theta": theta, "rho": rho}
