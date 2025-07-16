import numpy as np
from src.constants import OPTION_TYPES


class BinomialOption:
    
    def __init__(self, S, K, T, r, sigma, N, option_type="call", q=0.0):
        self.S = S
        self.K = K
        self.T = T
        self.r = r
        self.sigma = sigma
        self.N = N
        self.option_type = option_type
        self.q = q
        self.validate_inputs()


    def validate_inputs(self):
        if self.S <= 0 or self.K <= 0 or self.T <= 0 or self.sigma <= 0 or self.N <= 0:
            raise ValueError("All input parameters must be positive and non-zero.")
        if self.option_type not in OPTION_TYPES:
            raise ValueError("option_type must be either 'call' or 'put'.")


    def price_european(self, return_tree=False):
        dt = self.T / self.N
        u = np.exp(self.sigma * np.sqrt(dt))
        d = 1 / u
        discount = np.exp(-self.r * dt)
        p = (np.exp((self.r - self.q) * dt) - d) / (u - d)

        ST = np.array([self.S * (u ** (self.N - j)) * (d ** j) for j in range(self.N + 1)])

        if self.option_type == "call":
            option_values = np.maximum(ST - self.K, 0)
        else:
            option_values = np.maximum(self.K - ST, 0)

        for i in range(self.N - 1, -1, -1):
            option_values = discount * (p * option_values[:-1] + (1 - p) * option_values[1:])

        if return_tree:
            return option_values[0], ST
        return option_values[0]
    

    def price_american(self, return_tree=False):
        dt = self.T / self.N
        u = np.exp(self.sigma * np.sqrt(dt))
        d = 1 / u
        discount = np.exp(-self.r * dt)
        p = (np.exp((self.r - self.q) * dt) - d) / (u - d)

        asset_tree = np.zeros((self.N + 1, self.N + 1))
        for i in range(self.N + 1):
            for j in range(i + 1):
                asset_tree[j, i] = self.S * (u ** (i - j)) * (d ** j)

        option_tree = np.zeros_like(asset_tree)
        for j in range(self.N + 1):
            if self.option_type == "call":
                option_tree[j, self.N] = max(0, asset_tree[j, self.N] - self.K)
            else:
                option_tree[j, self.N] = max(0, self.K - asset_tree[j, self.N])

        for i in range(self.N - 1, -1, -1):
            for j in range(i + 1):
                continuation = discount * (p * option_tree[j, i + 1] + (1 - p) * option_tree[j + 1, i + 1])
                if self.option_type == "call":
                    exercise = max(0, asset_tree[j, i] - self.K)
                else:
                    exercise = max(0, self.K - asset_tree[j, i])
                option_tree[j, i] = max(continuation, exercise)

        if return_tree:
            return option_tree[0, 0], asset_tree
        return option_tree[0, 0]


    def price_and_tree(self, american=False):
        """
        Wrapper to compute either European or American option price and tree.
        """
        if american:
            return self.price_american(return_tree=True)
        else:
            return self.price_european(return_tree=True)