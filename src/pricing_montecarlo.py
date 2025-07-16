import numpy as np
from src.constants import (
    OPTION_TYPES, BARRIER_TYPES, STRIKE_TYPES, DEFAULT_DISCRETIZATION, DEFAULT_SIMULATIONS,
    DEFAULT_H, VALID_OPTION_TYPES, DEFAULT_PAYOUT, VALID_BARRIER_TYPES
)

class MonteCarloOption:
    def __init__(self, S, K, T, r, sigma, option_type="call", n_simulations=DEFAULT_SIMULATIONS, n_steps=DEFAULT_DISCRETIZATION, q=0.0):
        self.S = S
        self.K = K
        self.T = T
        self.r = r
        self.sigma = sigma
        self.option_type = option_type
        self.n_simulations = n_simulations
        self.n_steps = n_steps
        self.q = q
        self._validate_inputs()


    def _validate_inputs(self):
        if self.option_type not in OPTION_TYPES:
            raise ValueError("option_type must be 'call' or 'put'")
        if any(param <= 0 for param in [self.S, self.K, self.T, self.sigma, self.n_simulations, self.n_steps]):
            raise ValueError("All numeric inputs must be positive.")


    def price_asian(self):
        dt = self.T / self.n_steps
        drift = (self.r - 0.5 * self.sigma**2) * dt
        diffusion = self.sigma * np.sqrt(dt)
        discount = np.exp(-self.r * self.T)

        random_shocks = np.random.normal(0, 1, size=(self.n_simulations, self.n_steps))
        log_returns = drift + diffusion * random_shocks
        log_paths = np.cumsum(log_returns, axis=1)
        price_paths = self.S * np.exp(log_paths)
        avg_prices = price_paths.mean(axis=1)

        if self.option_type == "call":
            payoffs = np.maximum(avg_prices - self.K, 0)
        else:
            payoffs = np.maximum(self.K - avg_prices, 0)

        return discount * np.mean(payoffs)


    def price_asian_geometric(self):
        dt = self.T / self.n_steps
        drift = (self.r - 0.5 * self.sigma**2) * dt
        diffusion = self.sigma * np.sqrt(dt)
        discount = np.exp(-self.r * self.T)

        Z = np.random.normal(0, 1, size=(self.n_simulations, self.n_steps))
        log_returns = drift + diffusion * Z
        log_paths = np.cumsum(log_returns, axis=1)
        paths = self.S * np.exp(log_paths)

        geo_avg = np.exp(np.mean(np.log(paths), axis=1))
        if self.option_type == "call":
            payoffs = np.maximum(geo_avg - self.K, 0)
        else:
            payoffs = np.maximum(self.K - geo_avg, 0)

        return discount * np.mean(payoffs)


    def price_digital_barrier(self, barrier, barrier_type="up-and-in", payout=1.0):
        if barrier_type not in BARRIER_TYPES:
            raise ValueError("barrier_type must be one of " + str(BARRIER_TYPES))

        dt = self.T / self.n_steps
        discount = np.exp(-self.r * self.T)
        Z = np.random.normal(0, 1, (self.n_simulations, self.n_steps))
        paths = np.zeros_like(Z)
        paths[:, 0] = self.S

        for t in range(1, self.n_steps):
            paths[:, t] = paths[:, t-1] * np.exp((self.r - 0.5 * self.sigma**2) * dt + self.sigma * np.sqrt(dt) * Z[:, t])

        barrier_crossed = {
            "up-and-in":  np.any(paths >= barrier, axis=1),
            "up-and-out": np.all(paths <  barrier, axis=1),
            "down-and-in":  np.any(paths <= barrier, axis=1),
            "down-and-out": np.all(paths >  barrier, axis=1),
        }[barrier_type]

        ST = paths[:, -1]
        intrinsic = ST > self.K if self.option_type == "call" else ST < self.K

        final_payoff = payout * (barrier_crossed & intrinsic)
        return discount * np.mean(final_payoff)


    def price_lookback(self, strike_type="fixed"):
        if strike_type not in STRIKE_TYPES:
            raise ValueError("strike_type must be either 'fixed' or 'floating'")

        dt = self.T / self.n_steps
        discount = np.exp(-self.r * self.T)
        Z = np.random.normal(0, 1, size=(self.n_simulations, self.n_steps))
        paths = np.zeros_like(Z)
        paths[:, 0] = self.S

        drift = (self.r - 0.5 * self.sigma**2) * dt
        diffusion = self.sigma * np.sqrt(dt)
        for t in range(1, self.n_steps):
            paths[:, t] = paths[:, t - 1] * np.exp(drift + diffusion * Z[:, t])

        S_T = paths[:, -1]
        S_max = np.max(paths, axis=1)
        S_min = np.min(paths, axis=1)

        if strike_type == "fixed":
            payoffs = np.maximum(S_max - self.K, 0) if self.option_type == "call" else np.maximum(self.K - S_min, 0)
        else:
            payoffs = np.maximum(S_T - S_min, 0) if self.option_type == "call" else np.maximum(S_max - S_T, 0)

        return discount * np.mean(payoffs)


    def price_vanilla(self):
        dt = self.T / self.n_steps
        drift = (self.r - 0.5 * self.sigma**2) * dt
        diffusion = self.sigma * np.sqrt(dt)
        discount = np.exp(-self.r * self.T)

        Z = np.random.normal(0, 1, (self.n_simulations, self.n_steps))
        paths = np.zeros_like(Z)
        paths[:, 0] = self.S

        for t in range(1, self.n_steps):
            paths[:, t] = paths[:, t-1] * np.exp(drift + diffusion * Z[:, t])

        ST = paths[:, -1]
        payoffs = np.maximum(ST - self.K, 0) if self.option_type == "call" else np.maximum(self.K - ST, 0)

        return discount * np.mean(payoffs)


    def barrier_knock_in_out_payoff_paths(
            self,
            barrier,
            barrier_type="up-and-in",
            payout=DEFAULT_PAYOUT
        ):
            if self.option_type not in VALID_OPTION_TYPES:
                raise ValueError("option_type must be 'call' or 'put'")
            if barrier_type not in VALID_BARRIER_TYPES:
                raise ValueError("Invalid barrier type")

            dt = self.T / self.n_steps
            discount = np.exp(-self.r * self.T)

            Z = np.random.normal(0, 1, size=(self.n_simulations, self.n_steps))
            paths = np.zeros_like(Z)
            paths[:, 0] = self.S
            for t in range(1, self.n_steps):
                paths[:, t] = paths[:, t-1] * np.exp((self.r - 0.5 * self.sigma**2) * dt + self.sigma * np.sqrt(dt) * Z[:, t])

            ST = paths[:, -1]

            if self.option_type == "call":
                payoffs = ST > self.K
            else:
                payoffs = ST < self.K

            hit_barrier = {
                "up-and-in": np.any(paths >= barrier, axis=1),
                "up-and-out": np.all(paths < barrier, axis=1),
                "down-and-in": np.any(paths <= barrier, axis=1),
                "down-and-out": np.all(paths > barrier, axis=1),
            }[barrier_type]

            final_payoff = payout * (payoffs & hit_barrier)
            return discount * np.mean(final_payoff)


    def greek(self, greek, h=DEFAULT_H):
            if greek == "delta":
                price_up = MonteCarloOption(self.S + h, self.K, self.T, self.r, self.sigma, self.option_type, self.n_simulations, self.n_steps).price_vanilla()
                price_down = MonteCarloOption(self.S - h, self.K, self.T, self.r, self.sigma, self.option_type, self.n_simulations, self.n_steps).price_vanilla()
                return (price_up - price_down) / (2 * h)
            elif greek == "vega":
                price_up = MonteCarloOption(self.S, self.K, self.T, self.r, self.sigma + h, self.option_type, self.n_simulations, self.n_steps).price_vanilla()
                price_down = MonteCarloOption(self.S, self.K, self.T, self.r, self.sigma - h, self.option_type, self.n_simulations, self.n_steps).price_vanilla()
                return (price_up - price_down) / (2 * h)
            elif greek == "rho":
                price_up = MonteCarloOption(self.S, self.K, self.T, self.r + h, self.sigma, self.option_type, self.n_simulations, self.n_steps).price_vanilla()
                price_down = MonteCarloOption(self.S, self.K, self.T, self.r - h, self.sigma, self.option_type, self.n_simulations, self.n_steps).price_vanilla()
                return (price_up - price_down) / (2 * h)
            elif greek == "theta":
                price_up = MonteCarloOption(self.S, self.K, self.T + h, self.r, self.sigma, self.option_type, self.n_simulations, self.n_steps).price_vanilla()
                price_down = MonteCarloOption(self.S, self.K, self.T - h, self.r, self.sigma, self.option_type, self.n_simulations, self.n_steps).price_vanilla()
                return (price_up - price_down) / (2 * h)
            else:
                raise ValueError("Unsupported greek. Use 'delta', 'vega', 'rho', or 'theta'.")