# Utils Placeholder
import numpy as np
from src.utils.constants import VALID_OPTION_TYPES, VALID_BARRIER_TYPES, ERROR_OPTION_TYPE, ERROR_BARRIER_TYPE, ERROR_INVALID_INPUTS
import yfinance as yf
import pandas as pd


def validate_option_type(option_type):
    """
    Raises an error if the option type is invalid.
    """
    if option_type not in VALID_OPTION_TYPES:
        raise ValueError(ERROR_OPTION_TYPE)



def validate_barrier_type(barrier_type):
    """
    Raises an error if the barrier type is invalid.
    """
    if barrier_type not in VALID_BARRIER_TYPES:
        raise ValueError(ERROR_BARRIER_TYPE)



def validate_positive_inputs(*args):
    """
    Checks that all numeric inputs are strictly positive.
    """
    if any(param <= 0 for param in args):
        raise ValueError(ERROR_INVALID_INPUTS)



def calculate_payoff(ST, K, option_type):
    """
    Calculates the option payoff at maturity for European call or put.
    """
    if option_type == "call":
        return np.maximum(ST - K, 0)
    elif option_type == "put":
        return np.maximum(K - ST, 0)
    else:
        raise ValueError(ERROR_OPTION_TYPE)



def fetch_returns_from_yahoo(tickers: list, start: str, end: str) -> pd.DataFrame:
    """
        Fetches historical returns for given tickers from Yahoo Finance.
    """
    data = yf.download(tickers, start=start, end=end, auto_adjust=False, progress=False)

    # Handle multiple tickers (multi-index)
    if isinstance(data.columns, pd.MultiIndex):
        if "Adj Close" in data:
            prices = data["Adj Close"]
        elif "Close" in data:
            prices = data["Close"]
        else:
            raise ValueError("No 'Adj Close' or 'Close' found in data.")
    
    # Handle single ticker (flat dataframe)
    else:
        if "Adj Close" in data.columns:
            prices = data[["Adj Close"]].rename(columns={"Adj Close": tickers[0]})
        elif "Close" in data.columns:
            prices = data[["Close"]].rename(columns={"Close": tickers[0]})
        else:
            raise ValueError("No 'Adj Close' or 'Close' column found for single ticker.")

    if prices.empty or prices.isnull().all().all():
        raise ValueError("No valid price data found. Check the ticker symbols and date range.")

    returns = prices.pct_change().dropna()
    return returns