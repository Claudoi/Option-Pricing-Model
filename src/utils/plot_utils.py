import numpy as np
import pandas as pd
import plotly.graph_objs as go

from src.volatility.stochastic_volatility import HestonModel
from src.models.pricing_black_scholes import BlackScholesOption


def plot_black_scholes_heatmaps(K, T, r, q, S_min, S_max, sigma_min, sigma_max, resolution=50):
    S_vals = np.linspace(S_min, S_max, resolution)
    sigma_vals = np.linspace(sigma_min, sigma_max, resolution)
    S_grid, sigma_grid = np.meshgrid(S_vals, sigma_vals)

    price_call = np.zeros_like(S_grid)
    price_put = np.zeros_like(S_grid)

    for i in range(resolution):
        for j in range(resolution):
            S = S_grid[i, j]
            sigma = sigma_grid[i, j]
            call_opt = BlackScholesOption(S, K, T, r, sigma, option_type="call", q=q)
            put_opt = BlackScholesOption(S, K, T, r, sigma, option_type="put", q=q)
            price_call[i, j] = call_opt.price()
            price_put[i, j] = put_opt.price()

    fig_call = go.Figure(data=go.Heatmap(
        z=price_call,
        x=S_vals,
        y=sigma_vals,
        colorscale="Viridis",
        colorbar=dict(title="Call Price")
    ))
    fig_call.update_layout(
        title="Call Option Price Heatmap (Black-Scholes)",
        xaxis_title="Spot Price (S)",
        yaxis_title="Volatility (σ)",
        template="plotly_dark"
    )

    fig_put = go.Figure(data=go.Heatmap(
        z=price_put,
        x=S_vals,
        y=sigma_vals,
        colorscale="Viridis",
        colorbar=dict(title="Put Price")
    ))
    fig_put.update_layout(
        title="Put Option Price Heatmap (Black-Scholes)",
        xaxis_title="Spot Price (S)",
        yaxis_title="Volatility (σ)",
        template="plotly_dark"
    )

    return fig_call, fig_put


def plot_implied_volatility_vs_market_price(
    S: float,
    K: float,
    T: float,
    r: float,
    market_prices: np.ndarray,
    option_type: str = "call",
    q: float = 0.0
):
 
    implied_vols = []
    for price in market_prices:
        try:
            iv = BlackScholesOption.implied_volatility_newton(
                market_price=price,
                S=S,
                K=K,
                T=T,
                r=r,
                option_type=option_type,
                q=q
            )
            implied_vols.append(iv)
        except RuntimeError:
            implied_vols.append(np.nan)  # No converge

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=market_prices,
        y=implied_vols,
        mode="lines+markers",
        name="Implied Volatility"
    ))
    fig.update_layout(
        title="Implied Volatility vs Market Option Price",
        xaxis_title="Market Option Price",
        yaxis_title="Implied Volatility",
        template="plotly_dark"
    )
    return fig


def plot_price_vs_spot(K, T, r, sigma, option_type, q, model_class):
    S_range = np.linspace(50, 150, 100)
    prices = [model_class(S, K, T, r, sigma, option_type, q).price() for S in S_range]

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=S_range, y=prices, mode='lines', name='Option Price'))
    fig.update_layout(
        title="Option Price vs Spot Price",
        xaxis_title="Spot Price (S)",
        yaxis_title="Option Price",
        template="plotly_dark"
    )
    return fig


def plot_greeks_vs_spot(K, T, r, sigma, option_type, q, model_class):
    S_range = np.linspace(50, 150, 100)
    deltas, gammas = [], []

    for S in S_range:
        greeks = model_class(S, K, T, r, sigma, option_type, q).greeks()
        deltas.append(greeks["delta"])
        gammas.append(greeks["gamma"])

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=S_range, y=deltas, mode='lines', name='Delta'))
    fig.add_trace(go.Scatter(x=S_range, y=gammas, mode='lines', name='Gamma'))
    fig.update_layout(
        title="Delta & Gamma vs Spot Price",
        xaxis_title="Spot Price (S)",
        yaxis_title="Greek Value",
        template="plotly_dark"
    )
    return fig


def plot_market_vol_surface(vol_surface_obj, method: str = "linear"):
    vol_surface_obj.fetch_data()
    if vol_surface_obj.IV is None or len(vol_surface_obj.IV) == 0:
        raise ValueError("No implied volatility data found.")

    grid_K, grid_T, grid_IV = vol_surface_obj.interpolate(method=method)

    fig = go.Figure(data=[go.Surface(x=grid_K, y=grid_T, z=grid_IV)])
    fig.update_layout(
        title=f"Implied Volatility Surface for {vol_surface_obj.ticker}",
        scene=dict(
            xaxis_title="Strike (K)",
            yaxis_title="Maturity (T)",
            zaxis_title="Implied Volatility"
        ),
        template="plotly_dark",
        margin=dict(l=0, r=0, t=30, b=0)
    )
    return fig


def plot_svi_fit(log_moneyness, implied_vol, implied_vol_fit, T):
    k_sorted = np.argsort(log_moneyness)
    k = log_moneyness[k_sorted]
    iv_market = implied_vol[k_sorted]
    iv_model = implied_vol_fit[k_sorted]

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=k, y=iv_market, mode="markers", name="Market"))
    fig.add_trace(go.Scatter(x=k, y=iv_model, mode="lines", name="SVI Fit"))

    fig.update_layout(
        title=f"SVI Smile Fit (T = {T:.2f} years)",
        xaxis_title="Log-Moneyness (k = log(K/F))",
        yaxis_title="Implied Volatility",
        template="plotly_dark"
    )
    return fig


def plot_sabr_fit_surface(K: np.ndarray, market_vols: np.ndarray, sabr_vols: np.ndarray, F: float, T: float):
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=K,
        y=market_vols,
        mode="markers+lines",
        name="Market IV",
        marker=dict(color="royalblue", size=6)
    ))

    fig.add_trace(go.Scatter(
        x=K,
        y=sabr_vols,
        mode="lines",
        name="SABR Fit",
        line=dict(color="firebrick", width=2, dash="dash")
    ))

    fig.update_layout(
        title=f"SABR Volatility Fit (T={T:.2f} yrs, F={F})",
        xaxis_title="Strike (K)",
        yaxis_title="Implied Volatility",
        template="plotly_dark"
    )

    return fig


def plot_local_vol_surface(strikes: np.ndarray, maturities: np.ndarray, local_vol_grid: np.ndarray):
    T_mesh, K_mesh = np.meshgrid(maturities, strikes, indexing="ij")

    fig = go.Figure(data=[go.Surface(x=K_mesh, y=T_mesh, z=local_vol_grid)])
    fig.update_layout(
        title="Local Volatility Surface (Dupire Model)",
        scene=dict(
            xaxis_title="Strike (K)",
            yaxis_title="Maturity (T)",
            zaxis_title="Local Volatility"
        ),
        template="plotly_dark",
        margin=dict(l=0, r=0, t=30, b=0)
    )
    return fig


def plot_heston_price_vs_strike(S0, T, r, kappa, theta, sigma, rho, v0, option_type="call"):
    strikes = np.linspace(50, 150, 60)
    prices = []

    for K in strikes:
        model = HestonModel(S0, K, T, r, kappa, theta, sigma, rho, v0, option_type)
        prices.append(model.price())

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=strikes, y=prices, mode='lines', name='Heston Price'))
    fig.update_layout(
        title="Heston Model: Option Price vs Strike",
        xaxis_title="Strike Price (K)",
        yaxis_title="Option Price",
        template="plotly_dark"
    )
    return fig


def plot_rolling_var(returns: np.ndarray, var_series: np.ndarray, method: str = "ewma", confidence_level: float = 0.95, window: int = 100):
    returns = pd.Series(returns).dropna()
    x_range = range(window, window + len(var_series))

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=list(x_range),
        y=returns.iloc[window:].values,
        mode='lines',
        name='Returns',
        line=dict(color='steelblue')
    ))

    fig.add_trace(go.Scatter(
        x=list(x_range),
        y=-var_series,
        mode='lines',
        name=f'{int(confidence_level * 100)}% VaR',
        line=dict(color='firebrick')
    ))

    fig.update_layout(
        title=f"Rolling {method.upper()} VaR ({int(confidence_level * 100)}% Confidence Level)",
        xaxis_title="Time (Index)",
        yaxis_title="Return / VaR",
        template="plotly_dark",
        legend=dict(x=0, y=1.1, orientation="h")
    )
    return fig


def plot_var_es_histogram(returns: np.ndarray, var: float, es: float, title: str = "Portfolio Return Distribution"):
    fig = go.Figure()

    fig.add_trace(go.Histogram(
        x=returns,
        nbinsx=50,
        name="Returns",
        marker_color="skyblue",
        opacity=0.8
    ))

    fig.add_vline(x=-var, line=dict(color="red", dash="dash"), name="VaR")
    fig.add_vline(x=-es, line=dict(color="orange", dash="dash"), name="ES")

    fig.update_layout(
        title=title,
        xaxis_title="Return",
        yaxis_title="Frequency",
        template="plotly_dark",
        legend=dict(x=0.01, y=0.99)
    )
    return fig


def plot_garch_var_bar(var_value: float):
    fig = go.Figure()
    fig.add_trace(go.Bar(
        y=[var_value],
        name="GARCH VaR (1-day)",
        marker_color="firebrick"
    ))

    fig.update_layout(
        title="GARCH VaR (1-day ahead forecast)",
        yaxis_title="VaR",
        template="plotly_dark"
    )
    return fig


def plot_stress_testing_bar(stress_results: dict):
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=list(stress_results.keys()),
        y=list(stress_results.values()),
        name="Estimated Loss",
        marker_color="crimson"
    ))

    fig.update_layout(
        title="Stress Testing Results",
        xaxis_title="Scenario",
        yaxis_title="Estimated Portfolio Loss",
        template="plotly_dark"
    )
    return fig


def plot_risk_ratios_bar_chart(ratio_dict: dict):
    keys = []
    values = []
    colors = []

    for k, v in ratio_dict.items():
        if isinstance(v, (int, float)) and not np.isnan(v):
            keys.append(k)
            values.append(round(v, 4))

            if "Sharpe" in k or "Sortino" in k or "Information" in k or "Treynor" in k:
                colors.append("mediumseagreen")
            elif "Drawdown" in k or "VaR" in k or "Expected" in k:
                colors.append("indianred")
            elif "Skewness" in k or "Kurtosis" in k:
                colors.append("steelblue")
            else:
                colors.append("mediumpurple")

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=keys,
        y=values,
        marker_color=colors
    ))

    fig.update_layout(
        title="\ud83d\udcca Risk Ratios Overview",
        xaxis_title="Ratio",
        yaxis_title="Value",
        template="plotly_dark",
        xaxis_tickangle=30
    )
    return fig
