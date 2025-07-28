import numpy as np
from scipy.stats import norm


class DeltaHedgingSimulator:
    """
    Simulates a discrete-time delta hedging strategy using the Black-Scholes model.
    Computes the P&L of replicating an option by periodically rebalancing the portfolio.
    """

    def __init__(self, S0, K, T, r, sigma, option_type="call",
                 N_steps=50, N_paths=1000, hedge_freq=1):
        """
        Initializes the simulator with market and simulation parameters.

        Args:
            S0 (float): Initial spot price of the underlying asset.
            K (float): Strike price of the option.
            T (float): Time to maturity (in years).
            r (float): Risk-free interest rate.
            sigma (float): Volatility of the underlying asset.
            option_type (str): "call" or "put".
            N_steps (int): Number of time steps per simulated path.
            N_paths (int): Total number of simulated paths.
            hedge_freq (int): Frequency of delta rebalancing (every how many steps).
        """
        self.S0 = S0
        self.K = K
        self.T = T
        self.r = r
        self.sigma = sigma
        self.option_type = option_type.lower()
        self.N_steps = N_steps
        self.N_paths = N_paths
        self.hedge_freq = hedge_freq

        self.dt = T / N_steps
        self.time_grid = np.linspace(0, T, N_steps + 1)

    def _black_scholes_price(self, S, K, T, r, sigma, option_type):
        """
        Computes the price of a European option using the Black-Scholes formula.

        Returns:
            float: Option price.
        """
        if T <= 0:
            return max(S - K, 0) if option_type == "call" else max(K - S, 0)

        d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)

        if option_type == "call":
            return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
        else:
            return K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)

    def _delta_bs(self, S, K, T, r, sigma, option_type):
        """
        Computes the delta of a European option under Black-Scholes.

        Returns:
            float: Delta.
        """
        if T <= 0:
            return 1.0 if (option_type == "call" and S > K) else 0.0

        d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
        return norm.cdf(d1) if option_type == "call" else norm.cdf(d1) - 1


    def simulate(self):
        """
        Runs the Monte Carlo simulation of the delta hedging strategy.

        Returns:
            tuple: (pnl_paths, time_grid, pnl_over_time, hedging_errors)
                - pnl_paths: Array with final P&L per path.
                - time_grid: Time points of the simulation.
                - pnl_over_time: Array (N_paths, N_steps+1) with P&L over time.
                - hedging_errors: Array (N_paths, N_steps+1) with hedging error over time.
        """
        pnl_paths = []
        pnl_over_time = []
        hedging_errors = []

        for _ in range(self.N_paths):
            S = np.zeros(self.N_steps + 1)
            S[0] = self.S0

            for t in range(1, self.N_steps + 1):
                z = np.random.randn()
                S[t] = S[t - 1] * np.exp(
                    (self.r - 0.5 * self.sigma ** 2) * self.dt +
                    self.sigma * np.sqrt(self.dt) * z
                )

            cash_account = 0.0
            delta_prev = 0.0
            pnl_t = []
            error_t = []

            for t in range(0, self.N_steps, self.hedge_freq):
                T_remain = self.T - self.time_grid[t]
                if T_remain <= 0:
                    break

                S_t = S[t]
                delta = self._delta_bs(S_t, self.K, T_remain, self.r, self.sigma, self.option_type)
                d_delta = delta - delta_prev
                cash_account -= d_delta * S_t
                cash_account *= np.exp(self.r * self.dt * self.hedge_freq)
                delta_prev = delta

                portfolio = delta * S_t + cash_account
                option_val = self._black_scholes_price(S_t, self.K, T_remain, self.r, self.sigma, self.option_type)

                pnl_t.append(portfolio - option_val)
                error_t.append(abs(portfolio - option_val))  # <-- absolute hedging error

            payoff = self._black_scholes_price(S[-1], self.K, 0, self.r, self.sigma, self.option_type)
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
