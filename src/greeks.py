import numpy as np
from scipy.stats import norm
from src.constants import (OPTION_TYPES, ERROR_OPTION_TYPE, VEGA_SCALE, RHO_SCALE, THETA_SCALE)

class BlackScholesGreeks:
    def __init__(self, S, K, T, r, sigma, option_type='call'):
        self.S = S
        self.K = K
        self.T = T
        self.r = r
        self.sigma = sigma
        self.option_type = option_type
        self._validate_inputs()


    def _validate_inputs(self):
        if self.option_type not in OPTION_TYPES:
            raise ValueError(ERROR_OPTION_TYPE)
        if self.S <= 0 or self.K <= 0 or self.T <= 0 or self.sigma <= 0:
            raise ValueError("S, K, T, sigma must be positive and non-zero.")


    def _d1(self):
        return (np.log(self.S / self.K) + (self.r + 0.5 * self.sigma**2) * self.T) / (self.sigma * np.sqrt(self.T))


    def _d2(self):
        return self._d1() - self.sigma * np.sqrt(self.T)


    def delta(self):
        d1 = self._d1()
        return norm.cdf(d1) if self.option_type == 'call' else norm.cdf(d1) - 1


    def gamma(self):
        d1 = self._d1()
        return norm.pdf(d1) / (self.S * self.sigma * np.sqrt(self.T))


    def vega(self):
        d1 = self._d1()
        return (self.S * norm.pdf(d1) * np.sqrt(self.T)) * VEGA_SCALE


    def theta(self):
        d1 = self._d1()
        d2 = self._d2()
        term1 = -(self.S * norm.pdf(d1) * self.sigma) / (2 * np.sqrt(self.T))
        if self.option_type == 'call':
            term2 = self.r * self.K * np.exp(-self.r * self.T) * norm.cdf(d2)
            return (term1 - term2) * THETA_SCALE
        else:
            term2 = self.r * self.K * np.exp(-self.r * self.T) * norm.cdf(-d2)
            return (term1 + term2) * THETA_SCALE


    def rho(self):
        d2 = self._d2()
        factor = self.K * self.T * np.exp(-self.r * self.T)
        if self.option_type == 'call':
            return factor * norm.cdf(d2) * RHO_SCALE
        else:
            return -factor * norm.cdf(-d2) * RHO_SCALE