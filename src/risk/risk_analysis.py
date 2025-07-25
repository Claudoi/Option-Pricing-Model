import numpy as np
import pandas as pd
from scipy.stats import norm


from src.utils.constants import (
    DEFAULT_SIMULATIONS,
    ONE_OVER_100,
    EPSILON,
    ERROR_INVALID_INPUTS
)
from src.utils.utils import validate_positive_inputs


class PortfolioVaR:

    """
    Parametric (Variance-Covariance) VaR under normality.
    """

    def __init__(self, returns_df: pd.DataFrame, weights: np.ndarray, confidence_level: float = 0.95, holding_period: int = 1):
        validate_positive_inputs(confidence_level, holding_period)
        self.returns_df = returns_df
        self.weights = weights
        self.confidence_level = confidence_level
        self.holding_period = holding_period


    def calculate_var(self) -> float:
        mean_returns = self.returns_df.mean()
        cov_matrix = self.returns_df.cov()

        portfolio_mean = np.dot(self.weights, mean_returns)
        portfolio_std = np.sqrt(np.dot(self.weights.T, np.dot(cov_matrix, self.weights)))

        var = -portfolio_mean * self.holding_period + \
              portfolio_std * np.sqrt(self.holding_period) * norm.ppf(1 - self.confidence_level)
        return var
    
    def calculate_es(self) -> float:
        from scipy.stats import norm
        portfolio_mean = np.dot(self.weights, self.returns_df.mean())
        portfolio_std = np.sqrt(np.dot(self.weights.T, np.dot(self.returns_df.cov(), self.weights)))
        alpha = 1 - self.confidence_level
        es = -portfolio_mean + portfolio_std * norm.pdf(norm.ppf(alpha)) / alpha
        es *= self.holding_period ** 0.5
        return es



class HistoricalVaR:

    """
    Historical simulation-based VaR and Expected Shortfall (ES).
    """

    def __init__(self, portfolio_returns: np.ndarray, confidence_level: float = 0.95):
        validate_positive_inputs(confidence_level)
        self.returns = portfolio_returns
        self.confidence_level = confidence_level


    def calculate_var(self) -> float:
        var_percentile = 100 * (1 - self.confidence_level)
        return -np.percentile(self.returns, var_percentile)


    def calculate_es(self) -> float:
        threshold = -self.calculate_var()
        tail_losses = self.returns[self.returns < -threshold]
        return -np.mean(tail_losses) if len(tail_losses) > 0 else np.nan


class MonteCarloVaR:

    """
    Monte Carlo simulation-based VaR and Expected Shortfall (ES).
    """

    def __init__(self, S0: np.ndarray, mu: np.ndarray, sigma: np.ndarray, weights: np.ndarray,
                 T: float = 1.0, confidence_level: float = 0.95, n_sim: int = DEFAULT_SIMULATIONS):
        validate_positive_inputs(T, confidence_level, n_sim)
        self.S0 = S0
        self.mu = mu
        self.sigma = sigma
        self.weights = weights
        self.T = T
        self.confidence_level = confidence_level
        self.n_sim = n_sim


    def simulate_portfolio_returns(self) -> np.ndarray:
        n_assets = len(self.S0)
        Z = np.random.normal(0, 1, (self.n_sim, n_assets))
        drift = (self.mu - 0.5 * self.sigma ** 2) * self.T
        diffusion = self.sigma * np.sqrt(self.T) * Z
        ST = self.S0 * np.exp(drift + diffusion)
        portfolio_final = np.dot(ST, self.weights)
        portfolio_initial = np.dot(self.S0, self.weights)
        returns = (portfolio_final - portfolio_initial) / portfolio_initial
        return returns


    def calculate_var_es(self):
        simulated_returns = self.simulate_portfolio_returns()
        var_model = HistoricalVaR(simulated_returns, self.confidence_level)
        var = var_model.calculate_var()
        es = var_model.calculate_es()
        return var, es, simulated_returns



class StressTester:

    def __init__(self, prices_df: pd.DataFrame, weights: np.ndarray):
        self.prices = prices_df
        self.weights = weights

    def apply_shock(self, shock: dict) -> float:
        shocked_prices = self.prices.copy()
        for ticker, pct_change in shock.items():
            if ticker in shocked_prices.columns:
                shocked_prices[ticker] *= (1 + pct_change)

        returns = shocked_prices.pct_change().dropna()
        portfolio_returns = returns @ self.weights
        return portfolio_returns.mean(), portfolio_returns.std()
