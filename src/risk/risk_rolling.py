import numpy as np
import pandas as pd
from scipy.stats import norm
import logging

# Plotly for interactive charts
import plotly.graph_objs as go

# Configuración de logging
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

# Intentar importar arch_model (solo necesario si se usa GARCH)
try:
    from arch import arch_model
except ImportError:
    arch_model = None


class RollingVaR:
    """
    Rolling Value at Risk (VaR) estimator using EWMA or GARCH models.
    """

    def __init__(
        self,
        returns: np.ndarray,
        method: str = "ewma",
        lambda_: float = 0.94,
        window: int = 100,
        confidence_level: float = 0.95
    ):
        self.returns = pd.Series(returns).dropna()
        self.method = method.lower()
        self.lambda_ = lambda_
        self.window = window
        self.confidence_level = confidence_level

        if self.method not in {"ewma", "garch"}:
            raise ValueError("method must be either 'ewma' or 'garch'.")

        if self.method == "garch" and arch_model is None:
            raise ImportError("⚠️ 'arch' package is required for GARCH model.")

    def calculate_var_series(self):
        """
        Compute the rolling VaR series based on selected method.
        """
        z = abs(norm.ppf(1 - self.confidence_level))
        if self.method == "ewma":
            return self._calculate_ewma_var(z)
        elif self.method == "garch":
            return self._calculate_garch_var(z)

    def _calculate_ewma_var(self, z: float) -> np.ndarray:
        """
        EWMA-based rolling standard deviation for VaR estimation.
        """
        var_series = []
        for i in range(self.window, len(self.returns)):
            window_data = self.returns[i - self.window:i]
            weights = np.array([
                (1 - self.lambda_) * self.lambda_ ** (self.window - j - 1)
                for j in range(self.window)
            ])
            weights /= weights.sum()
            variance = np.dot(weights, (window_data - window_data.mean()) ** 2)
            ewma_std = np.sqrt(variance)
            var = z * ewma_std
            var_series.append(var)
        return np.array(var_series)

    def _calculate_garch_var(self, z: float) -> float:
        """
        GARCH(1,1)-based forecast of 1-day ahead VaR.
        """
        model = arch_model(self.returns * 100, vol="Garch", p=1, q=1)
        res = model.fit(disp="off")
        forecasts = res.forecast(horizon=1, reindex=False)
        forecast_var = forecasts.variance.values[-1, 0]
        var = z * np.sqrt(forecast_var) / 100  # Convert from % scale
        return var
