import numpy as np
import pandas as pd
from scipy.stats import norm
import logging

# Plotly for interactive charts
import plotly.graph_objs as go

# Logging setup
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

# Try to import arch_model (only needed if using GARCH)
try:
    from arch import arch_model
except ImportError:
    arch_model = None

    
class RollingVaR:
    """
    Rolling Value at Risk (VaR) estimator using EWMA or GARCH models.
    Always returns results as a NumPy array for consistency.
    """

    def __init__(self, returns: np.ndarray, method: str = "ewma",
                 lambda_: float = 0.94, window: int = 100,
                 confidence_level: float = 0.95):
        # Store returns as a float Series without NaNs
        self.returns = pd.Series(returns).dropna().astype(float)
        self.method = method.lower()
        self.lambda_ = float(lambda_)
        self.window = int(window)
        self.confidence_level = float(confidence_level)

        # Validate method name
        if self.method not in {"ewma", "garch"}:
            raise ValueError("method must be either 'ewma' or 'garch'.")

        # Ensure arch_model is available for GARCH method
        if self.method == "garch" and arch_model is None:
            raise ImportError("⚠️ 'arch' package is required for the GARCH model.")

    def calculate_var_series(self) -> np.ndarray:
        """
        Compute the VaR series based on the selected method.
        - EWMA: returns a rolling series (length = len(returns) - window)
        - GARCH: returns a single-element array with the 1-day ahead forecast
        """
        z = abs(norm.ppf(1 - self.confidence_level))  # Z-score for the given confidence level
        if self.method == "ewma":
            return self._calculate_ewma_var(z)
        else:  # GARCH
            return self._calculate_garch_var(z)

    def _calculate_ewma_var(self, z: float) -> np.ndarray:
        """
        EWMA-based rolling standard deviation for VaR estimation.
        Returns a NumPy array with one VaR value per rolling window.
        """
        if self.window >= len(self.returns):
            return np.array([], dtype=float)

        var_series = []

        # Fixed weights for the EWMA calculation
        weights = np.array([(1 - self.lambda_) * (self.lambda_ ** (self.window - j - 1))
                            for j in range(self.window)], dtype=float)
        weights /= weights.sum()  # Normalize weights

        # Rolling calculation
        for i in range(self.window, len(self.returns)):
            window_data = self.returns.iloc[i - self.window:i]
            variance = np.dot(weights, (window_data - window_data.mean())**2)
            ewma_std = np.sqrt(max(variance, 0.0))
            var = z * ewma_std
            var_series.append(var)

        return np.asarray(var_series, dtype=float)

    def _calculate_garch_var(self, z: float) -> np.ndarray:
        """
        GARCH(1,1)-based 1-day ahead VaR forecast.
        Returns a NumPy array with a single element.
        """
        # Fit the GARCH(1,1) model on returns (scaled to %)
        model = arch_model(self.returns * 100.0, vol="Garch", p=1, q=1)
        res = model.fit(disp="off")

        # Generate a 1-day ahead forecast
        forecasts = res.forecast(horizon=1, reindex=False)

        # Extract the variance forecast (flatten to scalar)
        forecast_var = float(np.asarray(forecasts.variance.values).reshape(-1)[-1])

        # Convert variance to standard deviation, then to VaR (back to unit scale)
        var = z * np.sqrt(max(forecast_var, 0.0)) / 100.0

        return np.array([var], dtype=float)