import numpy as np
from src.utils.utils import (
    validate_option_type,
    validate_positive_inputs,
    calculate_payoff
)


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
        validate_option_type(self.option_type)
        validate_positive_inputs(self.S, self.K, self.T, self.sigma, self.N)


    def price_european(self, return_tree=False):
        dt = self.T / self.N
        u = np.exp(self.sigma * np.sqrt(dt))
        d = 1 / u
        discount = np.exp(-self.r * dt)
        p = (np.exp((self.r - self.q) * dt) - d) / (u - d)

        ST = np.array([self.S * (u ** (self.N - j)) * (d ** j) for j in range(self.N + 1)])

        option_values = calculate_payoff(ST, self.K, self.option_type)

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
            option_tree[j, self.N] = calculate_payoff(asset_tree[j, self.N], self.K, self.option_type)

        for i in range(self.N - 1, -1, -1):
            for j in range(i + 1):
                continuation = discount * (p * option_tree[j, i + 1] + (1 - p) * option_tree[j + 1, i + 1])
                exercise = calculate_payoff(asset_tree[j, i], self.K, self.option_type)
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
        
        
    def get_tree(self):
        S = self.S
        u = np.exp(self.sigma * np.sqrt(self.T / self.N))
        d = 1 / u
        tree = []
        for i in range(self.N + 1):
            # En el paso i hay i precios "down" y N-i precios "up"
            level = [S * (u ** (i - j)) * (d ** j) for j in range(i + 1)]
            tree.append(level)
        return tree


    def get_sensitivities_tree(self, american=False):
        """
        Devuelve el árbol binomial con spot, valor de la opción, Delta y Gamma en cada nodo.
        Solo recomendable para N pequeño.
        """
        
        N = self.N
        dt = self.T / N
        u = np.exp(self.sigma * np.sqrt(dt))
        d = 1 / u
        discount = np.exp(-self.r * dt)
        p = (np.exp((self.r - self.q) * dt) - d) / (u - d)

        # Construcción de árbol de precios spot
        spot_tree = [[self.S * (u ** (i - j)) * (d ** j) for j in range(i + 1)] for i in range(N + 1)]
        # Construcción de árbol de valores de la opción
        value_tree = [[0.0 for _ in range(i + 1)] for i in range(N + 1)]
        # Inicializa hojas
        for j in range(N + 1):
            S_T = spot_tree[N][j]
            value_tree[N][j] = calculate_payoff(S_T, self.K, self.option_type)

        # Backward induction
        for i in range(N - 1, -1, -1):
            for j in range(i + 1):
                continuation = discount * (p * value_tree[i + 1][j] + (1 - p) * value_tree[i + 1][j + 1])
                S_ij = spot_tree[i][j]
                if american:
                    exercise = calculate_payoff(S_ij, self.K, self.option_type)
                    value_tree[i][j] = max(continuation, exercise)
                else:
                    value_tree[i][j] = continuation

        # Delta & Gamma
        node_tree = []
        for i in range(N):
            level = []
            for j in range(i + 1):
                S = spot_tree[i][j]
                V = value_tree[i][j]
                S_up = spot_tree[i + 1][j]
                S_down = spot_tree[i + 1][j + 1]
                V_up = value_tree[i + 1][j]
                V_down = value_tree[i + 1][j + 1]
                # Delta local
                delta = (V_up - V_down) / (S_up - S_down) if (S_up != S_down) else float('nan')
                # Gamma local (solo si no es penúltimo nivel)
                if i < N - 1:
                    S_uu = spot_tree[i + 2][j]
                    S_ud = spot_tree[i + 2][j + 1]
                    S_dd = spot_tree[i + 2][j + 2]
                    V_uu = value_tree[i + 2][j]
                    V_ud = value_tree[i + 2][j + 1]
                    V_dd = value_tree[i + 2][j + 2]
                    delta_up = (V_uu - V_ud) / (S_uu - S_ud) if (S_uu != S_ud) else float('nan')
                    delta_down = (V_ud - V_dd) / (S_ud - S_dd) if (S_ud != S_dd) else float('nan')
                    gamma = (delta_up - delta_down) / ((S_uu - S_dd) / 2) if (S_uu != S_dd) else float('nan')
                else:
                    gamma = float('nan')
                level.append({
                    'S': S,
                    'V': V,
                    'Delta': delta,
                    'Gamma': gamma
                })
            node_tree.append(level)
        # Añadir el último nivel (hojas, solo S y V)
        level = [{'S': spot_tree[N][j], 'V': value_tree[N][j], 'Delta': float('nan'), 'Gamma': float('nan')} for j in range(N + 1)]
        node_tree.append(level)
        return node_tree
