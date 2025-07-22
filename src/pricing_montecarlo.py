import numpy as np
from src.constants import (
    STRIKE_TYPES, DEFAULT_DISCRETIZATION, DEFAULT_SIMULATIONS,
    DEFAULT_H, DEFAULT_PAYOUT
)
from src.utils import (
    validate_option_type,
    validate_positive_inputs,
    validate_barrier_type,
    calculate_payoff
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
        validate_option_type(self.option_type)
        validate_positive_inputs(self.S, self.K, self.T, self.sigma, self.n_simulations, self.n_steps)

    def _simulate_paths(self):
        dt = self.T / self.n_steps
        Z = np.random.normal(0, 1, size=(self.n_simulations, self.n_steps))
        paths = np.zeros_like(Z)
        paths[:, 0] = self.S
        for t in range(1, self.n_steps):
            paths[:, t] = paths[:, t-1] * np.exp((self.r - 0.5 * self.sigma**2) * dt + self.sigma * np.sqrt(dt) * Z[:, t])
        return paths

    def price_asian(self):
        paths = self._simulate_paths()
        avg_prices = paths.mean(axis=1)
        discount = np.exp(-self.r * self.T)
        payoffs = calculate_payoff(avg_prices, self.K, self.option_type)
        return discount * np.mean(payoffs)

    def price_asian_geometric(self):
        paths = self._simulate_paths()
        geo_avg = np.exp(np.mean(np.log(paths), axis=1))
        discount = np.exp(-self.r * self.T)
        payoffs = calculate_payoff(geo_avg, self.K, self.option_type)
        return discount * np.mean(payoffs)

    def price_digital_barrier(self, barrier, barrier_type="up-and-in", payout=1.0):
        validate_barrier_type(barrier_type)
        paths = self._simulate_paths()
        ST = paths[:, -1]
        discount = np.exp(-self.r * self.T)

        barrier_crossed = {
            "up-and-in":  np.any(paths >= barrier, axis=1),
            "up-and-out": np.all(paths <  barrier, axis=1),
            "down-and-in":  np.any(paths <= barrier, axis=1),
            "down-and-out": np.all(paths >  barrier, axis=1),
        }[barrier_type]

        intrinsic = ST > self.K if self.option_type == "call" else ST < self.K
        final_payoff = payout * (barrier_crossed & intrinsic)
        return discount * np.mean(final_payoff)

    def price_lookback(self, strike_type="fixed"):
        if strike_type not in STRIKE_TYPES:
            raise ValueError("strike_type must be either 'fixed' or 'floating'")

        paths = self._simulate_paths()
        S_T = paths[:, -1]
        S_max = np.max(paths, axis=1)
        S_min = np.min(paths, axis=1)
        discount = np.exp(-self.r * self.T)

        if strike_type == "fixed":
            payoffs = np.maximum(S_max - self.K, 0) if self.option_type == "call" else np.maximum(self.K - S_min, 0)
        else:
            payoffs = np.maximum(S_T - S_min, 0) if self.option_type == "call" else np.maximum(S_max - S_T, 0)

        return discount * np.mean(payoffs)

    def price_vanilla(self):
        paths = self._simulate_paths()
        ST = paths[:, -1]
        discount = np.exp(-self.r * self.T)
        payoffs = calculate_payoff(ST, self.K, self.option_type)
        return discount * np.mean(payoffs)

    def barrier_knock_in_out_payoff_paths(self, barrier, barrier_type="up-and-in", payout=DEFAULT_PAYOUT):
        validate_option_type(self.option_type)
        validate_barrier_type(barrier_type)
        paths = self._simulate_paths()
        ST = paths[:, -1]
        discount = np.exp(-self.r * self.T)

        payoffs = ST > self.K if self.option_type == "call" else ST < self.K

        hit_barrier = {
            "up-and-in": np.any(paths >= barrier, axis=1),
            "up-and-out": np.all(paths < barrier, axis=1),
            "down-and-in": np.any(paths <= barrier, axis=1),
            "down-and-out": np.all(paths > barrier, axis=1),
        }[barrier_type]

        final_payoff = payout * (payoffs & hit_barrier)
        return discount * np.mean(final_payoff)
    
    def price_american_lsm(self, poly_degree=2):
        dt = self.T / self.n_steps
        discount = np.exp(-self.r * dt)
        paths = self._simulate_paths()
        
        if self.option_type == 'call':
            payoff = lambda S: np.maximum(S - self.K, 0)
        else:
            payoff = lambda S: np.maximum(self.K - S, 0)

        cashflows = payoff(paths[:, -1])
        
        for t in reversed(range(1, self.n_steps)):
            S_t = paths[:, t]
            in_the_money = payoff(S_t) > 0
            X = S_t[in_the_money]
            Y = cashflows[in_the_money] * discount
            
            if len(X) == 0:
                continue
            
            coeffs = np.polyfit(X, Y, deg=poly_degree)
            continuation_value = np.polyval(coeffs, X)
            
            exercise_value = payoff(X)
            exercise = exercise_value > continuation_value
            
            exercise_indices = np.where(in_the_money)[0][exercise]
            cashflows[exercise_indices] = exercise_value[exercise]
            cashflows[~np.isin(np.arange(len(paths)), exercise_indices)] *= discount

        return np.mean(cashflows) * np.exp(-self.r * dt)


    def greek(self, greek, h=DEFAULT_H):
        if greek == "delta":
            up = MonteCarloOption(self.S + h, self.K, self.T, self.r, self.sigma, self.option_type, self.n_simulations, self.n_steps).price_vanilla()
            down = MonteCarloOption(self.S - h, self.K, self.T, self.r, self.sigma, self.option_type, self.n_simulations, self.n_steps).price_vanilla()
        elif greek == "vega":
            up = MonteCarloOption(self.S, self.K, self.T, self.r, self.sigma + h, self.option_type, self.n_simulations, self.n_steps).price_vanilla()
            down = MonteCarloOption(self.S, self.K, self.T, self.r, self.sigma - h, self.option_type, self.n_simulations, self.n_steps).price_vanilla()
        elif greek == "rho":
            up = MonteCarloOption(self.S, self.K, self.T, self.r + h, self.sigma, self.option_type, self.n_simulations, self.n_steps).price_vanilla()
            down = MonteCarloOption(self.S, self.K, self.T, self.r - h, self.sigma, self.option_type, self.n_simulations, self.n_steps).price_vanilla()
        elif greek == "theta":
            up = MonteCarloOption(self.S, self.K, self.T + h, self.r, self.sigma, self.option_type, self.n_simulations, self.n_steps).price_vanilla()
            down = MonteCarloOption(self.S, self.K, self.T - h, self.r, self.sigma, self.option_type, self.n_simulations, self.n_steps).price_vanilla()
        else:
            raise ValueError("Unsupported greek. Use 'delta', 'vega', 'rho', or 'theta'.")
        return (up - down) / (2 * h)
