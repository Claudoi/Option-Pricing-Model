import numpy as np
from scipy.stats import norm
from src.models.pricing_black_scholes import BlackScholesOption



class ImpliedVolatility:
    """
    Implied volatility calculations for options.
    Provides methods to compute implied volatility using Newton-Raphson and bisection methods.
    """

    @staticmethod
    def validate_positive_inputs(*args):
        """
        Validate that all inputs are positive numbers.
        Raise ValueError if any input is not positive.
        """
        for val in args:
            if val <= 0:
                raise ValueError("All inputs must be positive and non-zero.")


    @staticmethod
    def vega(S, K, T, r, sigma, option_type="call", q=0.0):
        """
        Calculate Vega of an option for a given volatility sigma.
        Vega is the derivative of option price with respect to volatility.
        """
        opt = BlackScholesOption(S, K, T, r, sigma, option_type, q)
        d1 = opt._d1()
        return S * np.exp(-q * T) * norm.pdf(d1) * np.sqrt(T)


    @staticmethod
    def implied_volatility_newton(market_price, S, K, T, r, option_type="call", q=0.0,
        tol=1e-6, max_iter=100):
        """
        Compute implied volatility using Newton-Raphson method.
        """
        ImpliedVolatility.validate_positive_inputs(S, K, T, market_price)
        sigma = 0.2  # initial guess

        for i in range(max_iter):
            opt = BlackScholesOption(S, K, T, r, sigma, option_type, q)
            price = opt.price()
            v = ImpliedVolatility.vega(S, K, T, r, sigma, option_type, q)

            if v == 0:
                raise RuntimeError("Vega is zero. Newton-Raphson method fails.")

            price_diff = price - market_price
            if abs(price_diff) < tol:
                return sigma

            sigma -= price_diff / v

            # Clamp sigma to positive to avoid invalid values
            if sigma <= 0:
                sigma = tol

        raise RuntimeError("Implied volatility did not converge")


    @staticmethod
    def implied_volatility_bisection(market_price, S, K, T, r, option_type="call", q=0.0,
        tol=1e-6, max_iter=500, sigma_low=1e-6, sigma_high=5.0):
        """
        Compute implied volatility using the bisection method.
        """
        ImpliedVolatility.validate_positive_inputs(S, K, T, market_price)

        def price_diff(sigma):
            opt = BlackScholesOption(S, K, T, r, sigma, option_type, q)
            return opt.price() - market_price

        low = sigma_low
        high = sigma_high

        if price_diff(low) * price_diff(high) > 0:
            raise RuntimeError("Bisection method fails: f(low) and f(high) have same sign.")

        for i in range(max_iter):
            mid = (low + high) / 2
            mid_val = price_diff(mid)

            if abs(mid_val) < tol:
                return mid

            if price_diff(low) * mid_val < 0:
                high = mid
            else:
                low = mid

        raise RuntimeError("Implied volatility (bisection) did not converge")


    @staticmethod
    def implied_volatility_vectorized(market_prices, S, K, T, r, option_type="call", q=0.0,
        method="newton", tol=1e-6, max_iter=100):
        """
        Vectorized calculation of implied volatility for an array of market prices.
        """
        ivs = []
        for price in market_prices:
            try:
                if method == "newton":
                    iv = ImpliedVolatility.implied_volatility_newton(
                        price, S, K, T, r, option_type, q, tol, max_iter
                    )
                elif method == "bisection":
                    iv = ImpliedVolatility.implied_volatility_bisection(
                        price, S, K, T, r, option_type, q, tol=tol, max_iter=max_iter
                    )
                else:
                    raise ValueError(f"Unknown method '{method}'")
                ivs.append(iv)
            except Exception:
                ivs.append(np.nan)
        return np.array(ivs)
