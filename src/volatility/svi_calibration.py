import numpy as np
from scipy.optimize import minimize



class SVI_Calibrator:

    """    
    SVI volatility smile calibrator.
    Uses the SVI model to fit implied volatility surfaces.
    """

    def __init__(self, log_moneyness: np.ndarray, implied_vol: np.ndarray, maturity: float):
        self.k = log_moneyness
        self.iv = implied_vol
        self.T = maturity
        self.params_ = None
        self.implied_vol_fit_ = None


    @staticmethod
    def svi_total_variance(k, a, b, rho, m, sigma):
        """
        Computes the total variance according to the SVI model.
        """
        return a + b * (rho * (k - m) + np.sqrt((k - m) ** 2 + sigma ** 2))


    @staticmethod
    def objective_svi(params, k, market_variance):
        """
        Objective function for SVI calibration.
        """
        # Unpack parameters
        a, b, rho, m, sigma = params
        model = SVI_Calibrator.svi_total_variance(k, a, b, rho, m, sigma)
        return np.mean((model - market_variance) ** 2)


    def calibrate(self, x0=None, bounds=None):
        """
        Calibrates the SVI parameters to fit the implied volatility surface.
        Returns the fitted parameters and implied volatilities.
        """
        w_market = (self.iv ** 2) * self.T

        if x0 is None:
            x0 = [0.1, 0.1, 0.0, 0.0, 0.1]
        if bounds is None:
            bounds = [
                (-1, 1),     # a
                (0.001, 10), # b
                (-0.999, 0.999), # rho
                (-1, 1),     # m
                (0.001, 2)   # sigma
            ]

        result = minimize(
            SVI_Calibrator.objective_svi,
            x0,
            args=(self.k, w_market),
            bounds=bounds
        )

        if not result.success:
            raise ValueError("SVI calibration failed.")

        self.params_ = result.x
        w_fit = self.svi_total_variance(self.k, *self.params_)
        self.implied_vol_fit_ = np.sqrt(w_fit / self.T)
        return self.params_, self.implied_vol_fit_


    def get_params(self):
        if self.params_ is None:
            raise ValueError("Model not calibrated yet. Call calibrate() first.")
        return self.params_


    def get_fitted_vol(self):
        if self.implied_vol_fit_ is None:
            raise ValueError("Model not calibrated yet. Call calibrate() first.")
        return self.implied_vol_fit_


    @staticmethod
    def calibrate_svi_surface(k_matrix, iv_matrix, maturities):
        """
        Calibrates a SVI volatility surface for multiple maturities and strikes.
        Returns a 2D array of fitted volatilities and the SVI parameters.
        """
        vol_surface = []
        svi_params = []

        for i, T in enumerate(maturities):
            k = k_matrix[i]
            iv = iv_matrix[i]
            calibrator = SVI_Calibrator(k, iv, T)
            params, fitted_vols = calibrator.calibrate()
            vol_surface.append(fitted_vols)
            svi_params.append(params)

        return np.array(vol_surface), svi_params

