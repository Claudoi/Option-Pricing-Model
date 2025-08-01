import numpy as np
import pandas as pd
import yfinance as yf
from scipy.interpolate import griddata



class VolatilitySurface:
    
    """    
    Class to fetch and interpolate volatility surface data from Yahoo Finance.
    Supports multiple maturities and strikes.
    """

    def __init__(self, ticker: str, n_expirations: int = 5):
        self.ticker = ticker.upper()
        self.n_expirations = n_expirations
        self.K = None
        self.T = None
        self.IV = None
        self.grid_K = None
        self.grid_T = None
        self.grid_IV = None


    def fetch_data(self):
        """
        Fetches strike, maturity (in years), and implied volatilities from Yahoo Finance.
        Stores results as numpy arrays.
        """

        stock = yf.Ticker(self.ticker)
        expirations = stock.options[:self.n_expirations]

        K_vals, T_vals, IV_vals = [], [], []

        for exp in expirations:
            try:
                calls = stock.option_chain(exp).calls
                T = (pd.to_datetime(exp) - pd.Timestamp.today()).days / 365
                for _, row in calls.iterrows():
                    if pd.notnull(row["impliedVolatility"]):
                        K_vals.append(row["strike"])
                        T_vals.append(T)
                        IV_vals.append(row["impliedVolatility"])
            except Exception:
                continue

        if not IV_vals:
            raise ValueError("⚠️ No implied volatility data found.")

        self.K = np.array(K_vals)
        self.T = np.array(T_vals)
        self.IV = np.array(IV_vals)

        return self.K, self.T, self.IV


    def interpolate(self, resolution: int = 50, method: str = "linear"):
        """
        Interpolates the volatility surface onto a grid.
        Returns: meshgrid (K, T), interpolated IV values.
        """

        if self.K is None or self.T is None or self.IV is None:
            raise ValueError("Call fetch_data() before interpolate().")

        if method not in {"linear", "cubic", "nearest"}:
            raise ValueError(f"Interpolation method '{method}' is not supported.")

        grid_strike = np.linspace(min(self.K), max(self.K), resolution)
        grid_T = np.linspace(min(self.T), max(self.T), resolution)
        self.grid_K, self.grid_T = np.meshgrid(grid_strike, grid_T)

        self.grid_IV = griddata(
            (self.K, self.T), self.IV, (self.grid_K, self.grid_T), method=method
        )

        return self.grid_K, self.grid_T, self.grid_IV
