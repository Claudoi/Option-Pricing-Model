import numpy as np
import pandas as pd
import plotly.graph_objs as go
import streamlit as st


from src.volatility.stochastic_volatility import HestonModel
from src.models.pricing_black_scholes import BlackScholesOption
from src.models.implied_volatility import ImpliedVolatility

class PlotUtils:


    @staticmethod
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
            yaxis_title="Volatility (σ)"
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
            yaxis_title="Volatility (σ)"
        )

        return fig_call, fig_put



    @staticmethod
    def plot_binomial_price_vs_spot(K, T, r, sigma, N, option_type, q, BinomialOption):
        """
        Plot the Binomial option price vs spot price S for a given strike, time, rate, volatility, steps, and option type.
        """
        S_range = np.linspace(50, 150, 60)
        prices = []
        for S in S_range:
            binopt = BinomialOption(S, K, T, r, sigma, N, option_type, q)
            # By default, European. If you want American, add argument.
            price = binopt.price_european()
            prices.append(price)

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=S_range, y=prices, mode='lines', name='Binomial Price'))
        fig.update_layout(
            title="Binomial Option Price vs Spot Price",
            xaxis_title="Spot Price (S)",
            yaxis_title="Option Price"
        )
        return fig
    

    @staticmethod
    def show_binomial_tree(S, K, T, r, sigma, N, option_type, q, BinomialClass):
        import graphviz
        """
        Visualiza el árbol binomial de precios usando Graphviz.
        """
        option = BinomialClass(S, K, T, r, sigma, N, option_type, q)
        tree = option.get_tree()
        dot = graphviz.Digraph()

        # Añade nodos
        for i, level in enumerate(tree):
            for j, value in enumerate(level):
                node_id = f"{i}_{j}"
                dot.node(node_id, f"{value:.2f}")

        # Añade conexiones
        for i in range(len(tree) - 1):
            for j in range(len(tree[i])):
                dot.edge(f"{i}_{j}", f"{i+1}_{j}")
                dot.edge(f"{i}_{j}", f"{i+1}_{j+1}")

        st.graphviz_chart(dot)



    def graphviz_binomial_sensitivities(tree):
        import graphviz
        dot = graphviz.Digraph()
        N = len(tree) - 1
        for i, level in enumerate(tree):
            for j, node in enumerate(level):
                label = (
                    f"S={node['S']:.2f}\n"
                    f"V={node['V']:.2f}\n"
                    f"Δ={node['Delta']:.2f}\n"
                    f"Γ={node['Gamma']:.2f}"
                )
                name = f"n{i}_{j}"
                dot.node(name, label, shape="ellipse", style="filled", fillcolor="#f5f5f7")
                if i < N:
                    dot.edge(name, f"n{i+1}_{j}")
                    dot.edge(name, f"n{i+1}_{j+1}")
        return dot



    @staticmethod
    def plot_implied_volatility_vs_market_price_newton(S, K, T, r, market_prices, option_type="call", q=0.0):
        implied_vols = []
        for price in market_prices:
            try:
                iv = ImpliedVolatility.implied_volatility_newton(price, S, K, T, r, option_type, q)
                implied_vols.append(iv)
            except Exception:
                implied_vols.append(np.nan)

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=market_prices,
            y=implied_vols,
            mode="lines+markers",
            name="Implied Volatility (Newton)"
        ))
        fig.update_layout(
            title="Implied Volatility vs Market Option Price (Newton-Raphson)",
            xaxis_title="Market Option Price",
            yaxis_title="Implied Volatility",
            showlegend=True
        )
        return fig

    @staticmethod
    def plot_implied_volatility_vs_market_price_bisection(S, K, T, r, market_prices, option_type="call", q=0.0):
        implied_vols = []
        for price in market_prices:
            try:
                iv = ImpliedVolatility.implied_volatility_bisection(price, S, K, T, r, option_type, q)
                implied_vols.append(iv)
            except Exception:
                implied_vols.append(np.nan)

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=market_prices,
            y=implied_vols,
            mode="lines+markers",
            name="Implied Volatility (Bisection)"
        ))
        fig.update_layout(
            title="Implied Volatility vs Market Option Price (Bisection)",
            xaxis_title="Market Option Price",
            yaxis_title="Implied Volatility",
            showlegend=True
        )
        return fig

    @staticmethod
    def plot_implied_volatility_vs_market_price_vectorized(
        S, K, T, r, market_prices, option_type="call", q=0.0, method="newton"
    ):
        implied_vols = ImpliedVolatility.implied_volatility_vectorized(
            market_prices, S, K, T, r, option_type, q, method=method
        )

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=market_prices,
            y=implied_vols,
            mode="lines+markers",
            name=f"Implied Volatility ({method.capitalize()})"
        ))
        fig.update_layout(
            title=f"Implied Volatility vs Market Option Price ({method.capitalize()})",
            xaxis_title="Market Option Price",
            yaxis_title="Implied Volatility",
            showlegend=True
        )
        return fig


    @staticmethod
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
            yaxis_title="Greek Value"
        )
        return fig


    @staticmethod
    def plot_price_vs_spot(K, T, r, sigma, option_type, q, model_class):
        S_range = np.linspace(50, 150, 100)
        prices = []

        for S in S_range:
            option = model_class(S, K, T, r, sigma, option_type, q)
            prices.append(option.price())

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=S_range, y=prices, mode='lines', name='Option Price'))
        fig.update_layout(
            title=f"{option_type.capitalize()} Option Price vs Spot Price (Black-Scholes)",
            xaxis_title="Spot Price (S)",
            yaxis_title="Option Price"
        )
        return fig



    @staticmethod
    def plot_mc_paths(paths, n_show=50):
        """
        Plot simulated Monte Carlo price paths.
        """
        fig = go.Figure()
        for i in range(min(n_show, len(paths))):
            fig.add_trace(go.Scatter(
                y=paths[i],
                mode="lines",
                line=dict(width=1),
                showlegend=False,
                hoverinfo="skip"
            ))
        fig.update_layout(
            title="Monte Carlo Simulated Price Paths",
            xaxis_title="Step",
            yaxis_title="Price",
            plot_bgcolor="white"
        )
        return fig


    @staticmethod
    def plot_market_vol_surface(vol_surface_obj, method: str = "linear"):
        if vol_surface_obj.IV is None or len(vol_surface_obj.IV) == 0:
            raise ValueError("No implied volatility data found. Call fetch_data() first.")

        grid_K, grid_T, grid_IV = vol_surface_obj.interpolate(method=method)
        fig = go.Figure(data=[
            go.Surface(
                x=grid_K, y=grid_T, z=grid_IV,
                colorbar=dict(title="Implied Volatility", tickformat=".2%"),
                colorscale="Viridis",
                showscale=True,
                opacity=0.95
            )
        ])
        fig.update_layout(
            title=f"Implied Volatility Surface for {vol_surface_obj.ticker}",
            scene=dict(
                xaxis_title="Strike (K)",
                yaxis_title="Maturity (T, years)",
                zaxis_title="Implied Volatility",
                xaxis=dict(nticks=6, tickmode="auto"),
                yaxis=dict(nticks=6, tickmode="auto"),
                zaxis=dict(nticks=5, tickformat=".2%")
            ),
            autosize=True,
            margin=dict(l=10, r=10, t=40, b=10)
        )
        return fig

    @staticmethod
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
            yaxis_title="Implied Volatility"
        )
        return fig

    @staticmethod
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
            yaxis_title="Implied Volatility"
        )

        return fig


    @staticmethod
    def plot_local_vol_surface(strikes: np.ndarray, maturities: np.ndarray, local_vol_grid: np.ndarray):
        T_mesh, K_mesh = np.meshgrid(maturities, strikes, indexing="ij")  # (len(T), len(K)) para z-grid

        st.markdown("#### 🧪 Preview local volatility grid")
        st.dataframe(local_vol_grid)

        num_total = local_vol_grid.size
        num_valid = np.count_nonzero(~np.isnan(local_vol_grid))
        min_val = np.nanmin(local_vol_grid)
        max_val = np.nanmax(local_vol_grid)

        st.markdown(f"- 🔍 **Min:** {min_val:.4f} &nbsp;&nbsp;&nbsp; **Max:** {max_val:.4f}")
        st.markdown(f"- ✅ **Valores válidos:** {num_valid} / {num_total} ({100 * num_valid / num_total:.1f}%)")

        if np.isnan(local_vol_grid).all():
            st.error("⚠️ La superficie de volatilidad local es completamente NaN. Revisa los datos de IV o usa un grid más denso.")
            return None

        if num_valid < 0.3 * num_total:
            st.warning("⚠️ Menos del 30% de la superficie contiene valores válidos. El gráfico puede verse incompleto.")

        fig = go.Figure(data=[
            go.Surface(
                x=K_mesh,
                y=T_mesh,
                z=local_vol_grid,
                colorscale='Viridis',
                showscale=True,
                cmin=min_val,
                cmax=max_val
            )
        ])

        fig.update_layout(
            title="Local Volatility Surface (Dupire Model)",
            scene=dict(
                xaxis_title="Strike (K)",
                yaxis_title="Maturity (T)",
                zaxis_title="Local Volatility"
            ),
            margin=dict(l=0, r=0, t=30, b=0),
            height=600
        )

        return fig
       

    @staticmethod
    def plot_heston_price_vs_strike(S0, T, r, kappa, theta, sigma, rho, v0, option_type="call"):
        strikes = np.linspace(50, 150, 60)
        prices = []

        for K in strikes:
            try:
                model = HestonModel(S0, K, T, r, kappa, theta, sigma, rho, v0, option_type)
                price = model.price()
                prices.append(price)
            except Exception as e:
                prices.append(np.nan)  # En caso de error, mejor NaN que romper el gráfico

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=strikes,
            y=prices,
            mode='lines+markers',
            name='Heston Price',
            line=dict(color='royalblue'),
            hovertemplate="Strike: %{x:.2f}<br>Price: %{y:.2f}"
        ))

        fig.update_layout(
            title="📈 Heston Model: Option Price vs Strike",
            xaxis_title="Strike Price (K)",
            yaxis_title="Option Price",
            template="plotly_dark",
            height=500,
            margin=dict(l=40, r=40, t=60, b=40)
        )

        return fig



    @staticmethod
    def plot_heston_calibration_fit(market_data, S0, r, calibrated_params, option_type="call"):
        kappa, theta, sigma, rho, v0 = calibrated_params
        strikes = [d["K"] for d in market_data]
        maturities = [d["T"] for d in market_data]
        market_prices = [d["price"] for d in market_data]
        model_prices = []

        for d in market_data:
            model = HestonModel(S0, d["K"], d["T"], r, kappa, theta, sigma, rho, v0, option_type)
            model_prices.append(model.price())

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=strikes,
            y=market_prices,
            mode='markers',
            name='Market Prices',
            marker=dict(size=8, color='red'),
            hovertemplate="Strike: %{x}<br>Market: %{y:.2f}"
        ))

        fig.add_trace(go.Scatter(
            x=strikes,
            y=model_prices,
            mode='lines+markers',
            name='Calibrated Heston Model',
            marker=dict(size=6, color='green'),
            hovertemplate="Strike: %{x}<br>Model: %{y:.2f}"
        ))

        fig.update_layout(
            title="🎯 Heston Calibration: Market vs Model Prices",
            xaxis_title="Strike Price (K)",
            yaxis_title="Option Price",
            template="plotly_white",
            height=500,
            margin=dict(l=40, r=40, t=60, b=40)
        )

        return fig




    @staticmethod
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
            legend=dict(x=0, y=1.1, orientation="h")
        )
        return fig


    @staticmethod
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
            legend=dict(x=0.01, y=0.99)
        )
        return fig


    @staticmethod
    def plot_garch_var_bar(var_value: float):
        fig = go.Figure()
        fig.add_trace(go.Bar(
            y=[var_value],
            name="GARCH VaR (1-day)",
            marker_color="firebrick"
        ))

        fig.update_layout(
            title="GARCH VaR (1-day ahead forecast)",
            yaxis_title="VaR"
        )
        return fig


    @staticmethod
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
            yaxis_title="Estimated Portfolio Loss"
        )
        return fig


    @staticmethod
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
            title="📊 Risk Ratios Overview",
            xaxis_title="Ratio",
            yaxis_title="Value",
            xaxis_tickangle=30
        )
        return fig



    @staticmethod
    def plot_hedging_pnl(time_grid: np.ndarray, pnl: np.ndarray, title: str = "Delta Hedging P&L Over Time") -> go.Figure:
        
        pnl = np.atleast_1d(pnl)
        fig = go.Figure()

        # Main P&L line
        fig.add_trace(go.Scatter(
            x=time_grid,
            y=pnl,
            mode="lines+markers",
            name="Hedging P&L",
            line=dict(color="mediumseagreen", width=2),
            marker=dict(size=4),
            hovertemplate="Time: %{x:.2f}<br>P&L: %{y:.2f}<extra></extra>"
        ))

        # Reference line at P&L = 0
        fig.add_trace(go.Scatter(
            x=time_grid,
            y=np.zeros_like(time_grid),
            mode="lines",
            name="Break-even",
            line=dict(color="lightgray", dash="dash"),
            showlegend=True
        ))

        # Layout styling
        fig.update_layout(
            title=title,
            xaxis_title="Time to Maturity",
            yaxis_title="P&L",
            template="plotly_white",
            height=500,
            width=800,
            margin=dict(l=60, r=60, t=60, b=60),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5)
        )

        return fig
    

    @staticmethod
    def plot_hedging_pnl_histogram(pnl_paths: np.ndarray, title: str = "Histogram of Final P&L") -> go.Figure:


        # If pnl_paths is 2D (paths x time), take final value
        if pnl_paths.ndim == 2:
            pnl_final = pnl_paths[:, -1]
        else:
            pnl_final = pnl_paths

        fig = go.Figure()

        fig.add_trace(go.Histogram(
            x=pnl_final,
            nbinsx=50,
            marker_color="mediumseagreen",
            opacity=0.75,
            name="Final P&L"
        ))

        fig.update_layout(
            title=title,
            xaxis_title="Final P&L",
            yaxis_title="Frequency",
            template="plotly_white",
            height=450,
            width=750,
            margin=dict(l=50, r=50, t=60, b=50)
        )

        return fig

