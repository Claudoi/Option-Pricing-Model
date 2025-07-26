import numpy as np
from scipy.interpolate import CloughTocher2DInterpolator

class LocalVolatilitySurface:
    """
    Compute local volatility surface using Dupire's formula
    from a grid of implied volatilities.
    """

    def __init__(self, strikes: np.ndarray, maturities: np.ndarray, iv_surface: np.ndarray, F: float):
        self.strikes = strikes
        self.maturities = maturities
        self.iv_surface = iv_surface
        self.F = F

        # Mallado (K, T) para la superficie de volatilidad implícita
        K_mesh, T_mesh = np.meshgrid(strikes, maturities, indexing="xy")
        points = np.column_stack([K_mesh.flatten(), T_mesh.flatten()])
        values = iv_surface.flatten()

        # Interpolador robusto con extrapolación suave
        self.iv_interpolator = CloughTocher2DInterpolator(points, values)

    def _clip_inputs(self, K, T):
        K_clipped = np.clip(K, np.min(self.strikes), np.max(self.strikes))
        T_clipped = np.clip(T, np.min(self.maturities), np.max(self.maturities))
        return K_clipped, T_clipped

    def partial_derivatives(self, K, T, h=1e-2):
        K, T = self._clip_inputs(K, T)

        sigma = self.iv_interpolator(K, T)
        if sigma is None or sigma <= 0 or np.isnan(sigma):
            return np.nan, np.nan

        # Derivadas centradas
        sigma_T_plus = self.iv_interpolator(K, T + h)
        sigma_T_minus = self.iv_interpolator(K, T - h)
        sigma_K_plus = self.iv_interpolator(K + h, T)
        sigma_K_minus = self.iv_interpolator(K - h, T)

        # Evitar propagación de NaNs
        if np.any(np.isnan([sigma_T_plus, sigma_T_minus, sigma_K_plus, sigma_K_minus])):
            return np.nan, np.nan

        sigma_KK = sigma_K_plus - 2 * sigma + sigma_K_minus
        d_sigma_dT = (sigma_T_plus - sigma_T_minus) / (2 * h)
        d2_sigma_dK2 = sigma_KK / (h ** 2)

        return d_sigma_dT, d2_sigma_dK2

    def dupire_local_vol(self, K, T):
        K, T = self._clip_inputs(K, T)

        sigma = self.iv_interpolator(K, T)
        if sigma is None or sigma <= 0 or np.isnan(sigma):
            return np.nan

        d_sigma_dT, d2_sigma_dK2 = self.partial_derivatives(K, T)
        if np.isnan(d_sigma_dT) or np.isnan(d2_sigma_dK2):
            return np.nan

        try:
            # Fórmula clásica revisada de Dupire
            numerator = sigma**2 + 2 * T * sigma * d_sigma_dT + T * sigma**2 * d2_sigma_dK2
            denominator = (1 + K * d2_sigma_dK2)**2
            local_vol_squared = numerator / denominator if denominator != 0 else np.nan
            return np.sqrt(local_vol_squared) if local_vol_squared > 0 else np.nan
        except Exception:
            return np.nan

    def generate_surface(self):
        local_vol_grid = np.full((len(self.maturities), len(self.strikes)), np.nan)
        for i, T in enumerate(self.maturities):
            for j, K in enumerate(self.strikes):
                lv = self.dupire_local_vol(K, T)
                local_vol_grid[i, j] = lv
        return local_vol_grid
