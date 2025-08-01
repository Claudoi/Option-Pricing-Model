import numpy as np
import pandas as pd


class RiskRatios:
    """    
    Class to compute various risk ratios for a portfolio of returns.
    Includes Sharpe, Sortino, Calmar, Information, Treynor, Omega ratios,
    skewness, kurtosis, maximum drawdown, Value at Risk (VaR), and Expected Shortfall (ES).
    """

    def __init__(self, returns: np.ndarray, risk_free_rate: float = 0.0, benchmark_returns: np.ndarray = None):
        if isinstance(returns, pd.Series):
            returns = returns.values
        self.returns = returns
        self.risk_free_rate = risk_free_rate


        if benchmark_returns is not None:
            if isinstance(benchmark_returns, pd.Series):
                benchmark_returns = benchmark_returns.values
        self.benchmark_returns = benchmark_returns


    def sharpe_ratio(self) -> float:
        excess_returns = self.returns - self.risk_free_rate / 252
        return np.mean(excess_returns) / np.std(excess_returns, ddof=1)


    def sortino_ratio(self) -> float:
        excess_returns = self.returns - self.risk_free_rate / 252
        downside_std = np.std([r for r in excess_returns if r < 0], ddof=1)
        return np.mean(excess_returns) / downside_std if downside_std != 0 else np.nan


    def calmar_ratio(self) -> float:
        cumulative_returns = np.cumprod(1 + self.returns)
        peak = np.maximum.accumulate(cumulative_returns)
        drawdown = (cumulative_returns - peak) / peak
        max_drawdown = np.min(drawdown)
        annual_return = np.mean(self.returns) * 252
        return annual_return / abs(max_drawdown) if max_drawdown != 0 else np.nan


    def information_ratio(self) -> float:
        if self.benchmark_returns is None:
            return np.nan
        active_returns = self.returns - self.benchmark_returns
        return np.mean(active_returns) / np.std(active_returns, ddof=1)


    def treynor_ratio(self, beta: float) -> float:
        annual_return = np.mean(self.returns) * 252
        return (annual_return - self.risk_free_rate) / beta if beta != 0 else np.nan


    def omega_ratio(self, threshold: float = 0.0) -> float:
        gains = [r for r in self.returns if r > threshold]
        losses = [abs(r) for r in self.returns if r < threshold]
        return np.sum(gains) / np.sum(losses) if np.sum(losses) != 0 else np.nan


    def skewness(self) -> float:
        mean = np.mean(self.returns)
        std = np.std(self.returns, ddof=1)
        return np.mean(((self.returns - mean) / std) ** 3)


    def kurtosis(self) -> float:
        mean = np.mean(self.returns)
        std = np.std(self.returns, ddof=1)
        return np.mean(((self.returns - mean) / std) ** 4) - 3


    def max_drawdown(self) -> float:
        cumulative_returns = np.cumprod(1 + self.returns)
        peak = np.maximum.accumulate(cumulative_returns)
        drawdown = (cumulative_returns - peak) / peak
        return np.min(drawdown)


    def value_at_risk(self, alpha: float = 0.05) -> float:
        return -np.percentile(self.returns, 100 * alpha)


    def expected_shortfall(self, alpha: float = 0.05) -> float:
        var = self.value_at_risk(alpha)
        losses = self.returns[self.returns < -var]
        return -np.mean(losses) if len(losses) > 0 else np.nan
