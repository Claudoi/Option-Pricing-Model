import numpy as np
from scipy.stats import norm


class HestonDeltaHedgingSimulator:
    """
    Simulates a discrete-time delta hedging strategy using the Heston model.
    Underlying asset follows a stochastic volatility process.
    """

    def __init__(self, S0: float, K: float, T: float, r: float, v0: float, kappa: float, theta: float,
        sigma_v: float, rho: float, option_type: str = "call", N_steps: int = 50, N_paths: int = 1000,
        hedge_freq: int = 1, bump: float = 1e-4,
    ):
        self.S0 = S0
        self.K = K
        self.T = T
        self.r = r
        self.v0 = v0
        self.kappa = kappa
        self.theta = theta
        self.sigma_v = sigma_v
        self.rho = rho
        self.option_type = option_type.lower()
        self.N_steps = N_steps
        self.N_paths = N_paths
        self.hedge_freq = hedge_freq
        self.bump = bump  # For finite difference delta estimation

        self.dt = T / N_steps
        self.time_grid = np.linspace(0, T, N_steps + 1)



    def _black_scholes_price(self, S, K, T, r, sigma, option_type):
        if T <= 0:
            return max(S - K, 0) if option_type == "call" else max(K - S, 0)

        d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)

        if option_type == "call":
            return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
        else:
            return K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)



    def _finite_diff_delta(self, S, v, T_remain):
        """Approximate delta using central finite differences under BS with local vol sqrt(v)"""
        price_up = self._black_scholes_price(S + self.bump, self.K, T_remain, self.r, np.sqrt(v), self.option_type)
        price_down = self._black_scholes_price(S - self.bump, self.K, T_remain, self.r, np.sqrt(v), self.option_type)
        return (price_up - price_down) / (2 * self.bump)



    def simulate(self):
        pnl_paths = []
        pnl_over_time = []
        hedging_errors = []

        for _ in range(self.N_paths):
            S = np.zeros(self.N_steps + 1)
            v = np.zeros(self.N_steps + 1)
            S[0] = self.S0
            v[0] = self.v0

            for t in range(1, self.N_steps + 1):
                z1 = np.random.randn()
                z2 = np.random.randn()
                dw1 = z1
                dw2 = self.rho * z1 + np.sqrt(1 - self.rho**2) * z2

                v[t] = np.abs(v[t - 1] + self.kappa * (self.theta - v[t - 1]) * self.dt + self.sigma_v * np.sqrt(v[t - 1]) * np.sqrt(self.dt) * dw2)
                S[t] = S[t - 1] * np.exp((self.r - 0.5 * v[t - 1]) * self.dt + np.sqrt(v[t - 1]) * np.sqrt(self.dt) * dw1)

            cash_account = 0.0
            delta_prev = 0.0
            pnl_t = []
            error_t = []

            for t in range(0, self.N_steps, self.hedge_freq):
                T_remain = self.T - self.time_grid[t]
                if T_remain <= 0:
                    break

                S_t = S[t]
                v_t = v[t]
                delta = self._finite_diff_delta(S_t, v_t, T_remain)
                d_delta = delta - delta_prev
                cash_account -= d_delta * S_t
                cash_account *= np.exp(self.r * self.dt * self.hedge_freq)
                delta_prev = delta

                portfolio = delta * S_t + cash_account
                option_val = self._black_scholes_price(S_t, self.K, T_remain, self.r, np.sqrt(v_t), self.option_type)

                pnl_t.append(portfolio - option_val)
                error_t.append(abs(portfolio - option_val))

            payoff = self._black_scholes_price(S[-1], self.K, 0, self.r, np.sqrt(v[-1]), self.option_type)
            portfolio_final = delta_prev * S[-1] + cash_account
            pnl = portfolio_final - payoff

            pnl_paths.append(pnl)

            pnl_full = np.zeros(self.N_steps + 1)
            err_full = np.zeros(self.N_steps + 1)
            pnl_full[:len(pnl_t)] = pnl_t
            err_full[:len(error_t)] = error_t

            pnl_over_time.append(pnl_full)
            hedging_errors.append(err_full)

        return (
            np.array(pnl_paths),
            self.time_grid,
            np.array(pnl_over_time),
            np.array(hedging_errors)
        )
