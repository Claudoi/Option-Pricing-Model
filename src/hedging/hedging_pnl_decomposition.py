import numpy as np
from typing import Dict
from src.models.pricing_black_scholes import BlackScholesOption


class HedgingPnLAttribution:
    """
    Decomposes the P&L of a delta hedging strategy into:
    - Delta P&L: from changes in the underlying
    - Theta P&L: from passage of time
    - Residual P&L: due to discrete rehedging, approximation error, etc.
    """

    def __init__(
        self,
        S: np.ndarray,
        time_grid: np.ndarray,
        K: float,
        r: float,
        sigma: float,
        option_type: str = "call"
    ):
        self.S = S
        self.time_grid = time_grid
        self.T = time_grid[-1]
        self.K = K
        self.r = r
        self.sigma = sigma
        self.option_type = option_type.lower()
        self.dt = np.diff(time_grid)

        if self.option_type not in {"call", "put"}:
            raise ValueError("option_type must be 'call' or 'put'")



    def decompose(self) -> Dict[str, np.ndarray]:
        N = len(self.S) - 1
        delta_pnl = np.zeros(N)
        theta_pnl = np.zeros(N)
        approx_pnl = np.zeros(N)

        # Descomposición paso a paso
        for t in range(N):
            T_remain = self.T - self.time_grid[t]
            T_remain = max(T_remain, 1e-8)  # Para evitar T=0 en BS

            opt = BlackScholesOption(
                S=self.S[t],
                K=self.K,
                T=T_remain,
                r=self.r,
                sigma=self.sigma,
                option_type=self.option_type
            )
            greeks = opt.greeks()
            dS = self.S[t + 1] - self.S[t]
            dt = self.dt[t]

            delta_pnl[t] = greeks["delta"] * dS
            theta_pnl[t] = greeks["theta"] * dt
            approx_pnl[t] = delta_pnl[t] + theta_pnl[t]

        # Valor inicial (precio teórico al inicio)
        opt_start = BlackScholesOption(S=self.S[0], K=self.K, T=self.T, r=self.r, sigma=self.sigma, option_type=self.option_type)
        price_0 = opt_start.price()
        delta_0 = opt_start.greeks()["delta"]
        cash_0 = price_0 - delta_0 * self.S[0]

        # Payoff real al vencimiento
        ST = self.S[-1]
        payoff = max(ST - self.K, 0) if self.option_type == "call" else max(self.K - ST, 0)

        # Valor final de cartera si se mantiene delta constante desde t=0
        portfolio_T = delta_0 * ST + cash_0 * np.exp(self.r * self.T)

        # PnL realizado = valor final - payoff
        realized_pnl = portfolio_T - payoff

        # Residual = diferencia entre PnL real y aproximado
        residual_pnl = np.full(N, realized_pnl - np.sum(approx_pnl))

        return {
            "delta_pnl": delta_pnl,
            "theta_pnl": theta_pnl,
            "residual_pnl": residual_pnl,
            "total_pnl": approx_pnl + residual_pnl
        }
