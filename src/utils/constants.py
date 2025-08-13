import numpy as np


# --- Scaling Factors ---
VEGA_SCALE = 1/100     # Vega per 1% change in volatility
RHO_SCALE = 1/100      # Rho per 1% change in interest rate
THETA_SCALE = 1/365    # Theta per day


ONE_OVER_100 = 1/100
ONE_OVER_365 = 1/365

# --- stability thresholds ---
EPS_T = 1e-10       # Minimum time to maturity to avoid division by zero
EPS_SIG = 1e-12     # Minimum volatility to avoid division by zero


# --- Default Parameters ---
DEFAULT_N_STEPS = 100                # Time steps in simulation
DEFAULT_N_SIMULATIONS = 10000        # Monte Carlo paths
DEFAULT_DISCRETIZATION = 100         # Generic default for steps
DEFAULT_SIMULATIONS = 10000          # Alias for MC simulations
DEFAULT_H = 1e-2                     # Step size for finite differences (Greek estimation)
DEFAULT_PAYOUT = 1                # Default payout for digital options


EPSILON = DEFAULT_H                    # Alias for clarity


# --- Option, Barrier, and Strike Types ---
OPTION_TYPES = {"call", "put"}
VALID_OPTION_TYPES = OPTION_TYPES


BARRIER_TYPES = {"up-and-in", "up-and-out", "down-and-in", "down-and-out"}
VALID_BARRIER_TYPES = BARRIER_TYPES


STRIKE_TYPES = {"fixed", "floating"}


GREEK_TYPES = {"delta", "vega", "rho", "theta"}



# --- Error Messages ---
ERROR_OPTION_TYPE = "option_type must be either 'call' or 'put'."
ERROR_BARRIER_TYPE = "barrier_type must be one of 'up-and-in', 'up-and-out', 'down-and-in', 'down-and-out'."
ERROR_STRIKE_TYPE = "strike_type must be either 'fixed' or 'floating'."
ERROR_GREEK_TYPE = "Unsupported greek. Use 'delta', 'vega', 'rho', or 'theta'."
ERROR_NONPOSITIVE = "All numeric inputs must be positive and non-zero."
ERROR_INVALID_INPUTS = "S, K, T and sigma must be positive and non-zero."



# --- Barrier Logic Map ---
BARRIER_CROSSED_LOGIC = {
    "up-and-in":     lambda paths, barrier: np.any(paths >= barrier, axis=1),
    "up-and-out":    lambda paths, barrier: np.all(paths <  barrier, axis=1),
    "down-and-in":   lambda paths, barrier: np.any(paths <= barrier, axis=1),
    "down-and-out":  lambda paths, barrier: np.all(paths >  barrier, axis=1),
}
