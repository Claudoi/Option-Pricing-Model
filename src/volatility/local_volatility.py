import numpy as np
from scipy.interpolate import LinearNDInterpolator

class LocalVolatilitySurface:
    """
    Compute local volatility surface using Dupire's formula
    from a grid of implied volatilities.
    """

    def __init__(self, strikes: np.ndarray, maturities: np.ndarray, iv_surface: np.ndarray, F: float):
        """
        Parameters:
            strikes: 1D array of strikes
            maturities: 1D array of maturities (in years)
            iv_surface: 2D array (len(T) x len(K)) of implied volatilities
            F: current forward price
        """
        self.strikes = strikes
        self.maturities = maturities
        self.iv_surface = iv_surface
        self.F = F

        # Create interpolator for IV
        T_mesh, K_mesh = np.meshgrid(maturities, strikes, indexing="ij")
        self.iv_interpolator = LinearNDInterpolator(list(zip(K_mesh.flatten(), T_mesh.flatten())), iv_surface.flatten())

    def partial_derivatives(self, K, T, h=1e-4):
        """
        Estimate ∂σ/∂T and ∂²σ/∂K² using finite differences.
        """
        sigma = self.iv_interpolator(K, T)
        sigma_T_plus = self.iv_interpolator(K, T + h)
        sigma_T_minus = self.iv_interpolator(K, T - h)
        sigma_K_plus = self.iv_interpolator(K + h, T)
        sigma_K_minus = self.iv_interpolator(K - h, T)
        sigma_KK = self.iv_interpolator(K + h, T) - 2 * sigma + self.iv_interpolator(K - h, T)

        d_sigma_dT = (sigma_T_plus - sigma_T_minus) / (2 * h)
        d2_sigma_dK2 = sigma_KK / (h ** 2)

        return d_sigma_dT, d2_sigma_dK2

    def dupire_local_vol(self, K, T):
        """
        Compute local volatility using Dupire's formula.
        """
        sigma = self.iv_interpolator(K, T)
        if sigma is None or sigma <= 0:
            return np.nan

        d_sigma_dT, d2_sigma_dK2 = self.partial_derivatives(K, T)

        numerator = d_sigma_dT + 2 * sigma * T * d_sigma_dT
        denominator = (1 + K * sigma * d2_sigma_dK2 / sigma) ** 2

        local_vol_squared = numerator / (T * denominator) if denominator != 0 else np.nan
        return np.sqrt(local_vol_squared) if local_vol_squared > 0 else np.nan

    def generate_surface(self):
        """
        Generate full local volatility surface as a grid.
        """
        local_vol_grid = np.zeros_like(self.iv_surface)
        for i, T in enumerate(self.maturities):
            for j, K in enumerate(self.strikes):
                local_vol_grid[i, j] = self.dupire_local_vol(K, T)
        return local_vol_grid
