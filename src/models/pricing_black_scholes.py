import numpy as np
from scipy.stats import norm
from src.utils.constants import ONE_OVER_100, ONE_OVER_365
from src.utils.utils import validate_option_type, validate_positive_inputs



class BlackScholesOption:
    """
    Black-Scholes option pricing model.
    Computes the price and greeks for European call and put options.
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
        return (np.log(self.S / self.K) + (self.r - self.q + 0.5 * self.sigma ** 2) * self.T) / (self.sigma * np.sqrt(self.T))


    def _d2(self):

        return self._d1() - self.sigma * np.sqrt(self.T)


    def price(self):
        """
        Computes the Black-Scholes option price.
        Returns the price of the option.
        """
        d1 = self._d1()
        d2 = self._d2()
        discounted_S = self.S * np.exp(-self.q * self.T)
        if self.option_type == "call":
            return discounted_S * norm.cdf(d1) - self.K * np.exp(-self.r * self.T) * norm.cdf(d2)
        else:
            return self.K * np.exp(-self.r * self.T) * norm.cdf(-d2) - discounted_S * norm.cdf(-d1)


    def greeks(self):
        """        
        Computes the option greeks: delta, gamma, vega, theta, rho.
        Returns a dictionary with the computed values.
        """
        d1 = self._d1()
        d2 = self._d2()
        sqrt_T = np.sqrt(self.T)
        discounted_S = self.S * np.exp(-self.q * self.T)
        price = self.price()

        delta = (
            np.exp(-self.q * self.T) * norm.cdf(d1)
            if self.option_type == "call"
            else np.exp(-self.q * self.T) * (norm.cdf(d1) - 1)
        )

        gamma = np.exp(-self.q * self.T) * norm.pdf(d1) / (self.S * self.sigma * sqrt_T)
        
        vega = self.S * np.exp(-self.q * self.T) * norm.pdf(d1) * sqrt_T * ONE_OVER_100
        
        theta = (
            (-self.S * norm.pdf(d1) * self.sigma * np.exp(-self.q * self.T) / (2 * sqrt_T)
             - self.r * self.K * np.exp(-self.r * self.T) * norm.cdf(d2)) * ONE_OVER_365
            if self.option_type == "call"
            else (-self.S * norm.pdf(d1) * self.sigma * np.exp(-self.q * self.T) / (2 * sqrt_T)
                  + self.r * self.K * np.exp(-self.r * self.T) * norm.cdf(-d2)) * ONE_OVER_365
        )

        rho = (
            self.K * self.T * np.exp(-self.r * self.T) * norm.cdf(d2) * ONE_OVER_100
            if self.option_type == "call"
            else -self.K * self.T * np.exp(-self.r * self.T) * norm.cdf(-d2) * ONE_OVER_100
        )

        return {
            "price": price,
            "delta": delta,
            "gamma": gamma,
            "vega": vega,
            "theta": theta,
            "rho": rho
        }
