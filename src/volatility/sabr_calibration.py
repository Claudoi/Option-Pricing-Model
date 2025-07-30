import numpy as np
from scipy.optimize import minimize

class SABRCalibrator:
    """
    SABR volatility smile calibrator. 
    Usa la aproximación de Hagan para volatilidad implícita.
    """
    def __init__(self, F, K, T, market_vols, beta_fixed=0.5):
        """
        F: Forward price
        K: Array de strikes
        T: Maturity (en años)
        market_vols: Array de volatilidades implícitas del mercado
        beta_fixed: Exponente beta (normalmente 0.5 o 1.0)
        """
        self.F = F
        self.K = np.array(K)
        self.T = T
        self.market_vols = np.array(market_vols)
        self.beta_fixed = beta_fixed
        self.params = None  # alpha, beta, rho, nu


    @staticmethod
    def sabr_volatility(F, K, T, alpha, beta, rho, nu):
        """Hagan's SABR approximation."""
        if F == K:
            numer = alpha
            denom = F**(1 - beta)
            term1 = ((1 - beta)**2 / 24) * (alpha**2) / (F**(2 - 2 * beta))
            term2 = (rho * beta * nu * alpha) / (4 * F**(1 - beta))
            term3 = ((2 - 3 * rho**2) * nu**2) / 24
            return (alpha / denom) * (1 + (term1 + term2 + term3) * T)
        else:
            logFK = np.log(F / K)
            z = (nu / alpha) * (F * K)**((1 - beta) / 2) * logFK
            x_z = np.log((np.sqrt(1 - 2 * rho * z + z**2) + z - rho) / (1 - rho))
            numer = alpha * z
            denom = (F * K)**((1 - beta) / 2) * (1 + ((1 - beta)**2 / 24) * logFK**2 + ((1 - beta)**4 / 1920) * logFK**4)
            term1 = 1 + (((1 - beta)**2 / 24) * (alpha**2) / ((F * K)**(1 - beta))
                        + (rho * beta * nu * alpha) / (4 * (F * K)**((1 - beta) / 2))
                        + ((2 - 3 * rho**2) * nu**2) / 24) * T
            return (numer / denom) * (z / x_z) * term1


    @staticmethod
    def _objective(params, F, K, T, market_vols):
        alpha, beta, rho, nu = params
        model_vols = [SABRCalibrator.sabr_volatility(F, k, T, alpha, beta, rho, nu) for k in K]
        return np.mean((np.array(model_vols) - market_vols) ** 2)


    def calibrate(self):
        """
        Calibra los parámetros del modelo SABR.
        Devuelve: params (alpha, beta, rho, nu)
        """
        x0 = [0.2, self.beta_fixed, 0.0, 0.5]  # alpha, beta, rho, nu
        bounds = [
            (0.001, 2.0),    # alpha
            (self.beta_fixed, self.beta_fixed),  # beta fixed
            (-0.999, 0.999), # rho
            (0.001, 2.0)     # nu
        ]
        result = minimize(
            SABRCalibrator._objective, x0,
            args=(self.F, self.K, self.T, self.market_vols),
            bounds=bounds
        )
        if not result.success:
            raise ValueError("SABR calibration failed.")
        self.params = result.x
        return self.params


    def model_vols(self):
        """
        Devuelve las volatilidades ajustadas con los parámetros calibrados.
        """
        if self.params is None:
            raise RuntimeError("Call calibrate() first.")
        alpha, beta, rho, nu = self.params
        return np.array([
            SABRCalibrator.sabr_volatility(self.F, k, self.T, alpha, beta, rho, nu)
            for k in self.K
        ])


    @staticmethod
    def calibrate_sabr_surface(strike_matrix, iv_matrix, maturities, forward_price, beta=0.5):
        vol_surface = []
        sabr_params = []

        for i, T in enumerate(maturities):
            strikes = strike_matrix[i]
            ivs = iv_matrix[i]
            calibrator = SABRCalibrator(forward_price, strikes, T, ivs, beta_fixed=beta)
            params = calibrator.calibrate()
            fitted_vols = calibrator.model_vols()
            vol_surface.append(fitted_vols)
            sabr_params.append(params)

        return np.array(vol_surface), sabr_params

