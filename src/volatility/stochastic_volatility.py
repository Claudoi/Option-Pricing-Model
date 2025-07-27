import numpy as np
from scipy.integrate import quad
from numpy import log, exp, sqrt, pi


class HestonModel:
    def __init__(self, S0, K, T, r, kappa, theta, sigma, rho, v0, option_type="call"):
        self.S0 = S0
        self.K = K
        self.T = T
        self.r = r
        self.kappa = kappa
        self.theta = theta
        self.sigma = sigma
        self.rho = rho
        self.v0 = v0
        self.option_type = option_type.lower()

    def _char_func(self, phi, Pnum):
        i = 1j
        u = 0.5 if Pnum == 1 else -0.5
        b = self.kappa - self.rho * self.sigma if Pnum == 1 else self.kappa
        a = self.kappa * self.theta

        d = np.sqrt((self.rho * self.sigma * i * phi - b) ** 2 - (self.sigma ** 2) * (2 * u * i * phi - phi ** 2))
        g = (b - self.rho * self.sigma * i * phi + d) / (b - self.rho * self.sigma * i * phi - d)

        C = self.r * i * phi * self.T + (a / self.sigma ** 2) * (
            (b - self.rho * self.sigma * i * phi + d) * self.T
            - 2 * np.log((1 - g * np.exp(d * self.T)) / (1 - g))
        )

        D = ((b - self.rho * self.sigma * i * phi + d) / self.sigma ** 2) * (
            (1 - np.exp(d * self.T)) / (1 - g * np.exp(d * self.T))
        )

        return np.exp(C + D * self.v0 + i * phi * np.log(self.S0))

    def _integrand(self, phi, Pnum):
        if phi == 0:
            return 0.0  # evitar división por cero
        return np.real(np.exp(-1j * phi * np.log(self.K)) * self._char_func(phi, Pnum) / (1j * phi))

    def price(self):
        # Evitamos phi = 0 en la integración
        integral1 = quad(lambda phi: self._integrand(phi, 1), 1e-5, 100, limit=100)[0]
        integral2 = quad(lambda phi: self._integrand(phi, 2), 1e-5, 100, limit=100)[0]

        P1 = 0.5 + (1 / pi) * integral1
        P2 = 0.5 + (1 / pi) * integral2

        call_price = self.S0 * P1 - self.K * exp(-self.r * self.T) * P2

        if self.option_type == "call":
            return call_price
        elif self.option_type == "put":
            return call_price - self.S0 + self.K * exp(-self.r * self.T)
        else:
            raise ValueError("option_type must be 'call' or 'put'")
