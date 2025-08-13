import numpy as np
import pandas as pd
import plotly.graph_objs as go
import streamlit as st


from src.volatility.stochastic_volatility import HestonModel
from src.models.pricing_black_scholes import bs_price_vectorized, bs_greeks_vectorized  
from src.models.implied_volatility import ImpliedVolatility

class PlotUtils:


     
    @staticmethod
    def _colors_for_metric(metric: str):
        metric = metric.lower()
        if metric in ("delta", "theta", "rho"):
            return "RdBu", 0.0      
 
        return "Cividis", None      


    @staticmethod
    def plot_black_scholes_heatmaps(K, T, r, q, S_min, S_max, sigma_min, sigma_max, resolution=50):
        S_vals = np.linspace(S_min, S_max, resolution)
        sigma_vals = np.linspace(sigma_min, sigma_max, resolution)
        S_grid, sigma_grid = np.meshgrid(S_vals, sigma_vals)

        price_call = bs_price_vectorized(S_grid, K, T, r, sigma_grid, option_type="call", q=q)
        price_put  = bs_price_vectorized(S_grid, K, T, r, sigma_grid, option_type="put",  q=q)


        colorscale = "Cividis"

        fig_call = go.Figure(data=go.Heatmap(
            z=price_call, x=S_vals, y=sigma_vals,
            colorscale=colorscale,
            colorbar=dict(title="Call Price"),
            hovertemplate="S=%{x:.4g}<br>σ=%{y:.4f}<br>Price=%{z:.6f}<extra></extra>"
        ))
        fig_call.update_layout(
            title="Call Option Price Heatmap (Black-Scholes)",
            xaxis_title="Spot Price (S)",
            yaxis_title="Volatility (σ)",
            height=500
        )

        fig_put = go.Figure(data=go.Heatmap(
            z=price_put, x=S_vals, y=sigma_vals,
            colorscale=colorscale,
            colorbar=dict(title="Put Price"),
            hovertemplate="S=%{x:.4g}<br>σ=%{y:.4f}<br>Price=%{z:.6f}<extra></extra>"
        ))
        fig_put.update_layout(
            title="Put Option Price Heatmap (Black-Scholes)",
            xaxis_title="Spot Price (S)",
            yaxis_title="Volatility (σ)",
            height=500
        )

        return fig_call, fig_put


    @staticmethod
    def plot_bs_heatmap_flexible(
        *,
        axes: str,                  # "S-sigma" | "K-T"
        metric: str,                # "price"|"delta"|"gamma"|"vega"|"theta"|"rho"
        option_type: str,           # "call"|"put"
        S: float, K: float, T: float, r: float, q: float, sigma: float,
        S_min=None, S_max=None, sigma_min=None, sigma_max=None,
        K_min=None, K_max=None, T_min=None, T_max=None,
        resolution: int = 50,
        colorscale: str | None = None,   # override palette if provided
        zmid_override: float | None = None  # override zmid if provided
    ):

        default_colorscale, default_zmid = PlotUtils._colors_for_metric(metric)
        cs = colorscale or default_colorscale
        zmid_val = default_zmid if zmid_override is None else zmid_override

        if axes == "S-sigma":
            x_vals = np.linspace(S_min, S_max, resolution)
            y_vals = np.linspace(sigma_min, sigma_max, resolution)
            X, Y = np.meshgrid(x_vals, y_vals)  # rows=y (sigma), cols=x (S)

            if metric == "price":
                Z = bs_price_vectorized(X, K, T, r, Y, option_type=option_type, q=q)
                cbar = f"{option_type.capitalize()} Price"
                # price is non-negative → ensure sequential palette
                if colorscale is None:
                    cs, zmid_val = "Cividis", None
            else:
                G = bs_greeks_vectorized(X, K, T, r, Y, option_type=option_type, q=q)
                Z = G[metric]
                cbar = metric.capitalize()

            fig = go.Figure(data=go.Heatmap(
                z=Z, x=x_vals, y=y_vals,
                colorscale=cs,
                zmid=zmid_val,  # will be ignored if None
                colorbar=dict(title=cbar),
                hovertemplate="S=%{x:.4g}<br>σ=%{y:.4f}<br>"+cbar+"=%{z:.6f}<extra></extra>"
            ))
            fig.update_layout(
                title=f"Black-Scholes {cbar} Heatmap — axes: S vs σ",
                xaxis_title="Spot (S)",
                yaxis_title="Volatility (σ)",
                height=500
            )
            return fig

        elif axes == "K-T":
            x_vals = np.linspace(K_min, K_max, resolution)
            y_vals = np.linspace(T_min, T_max, resolution)
            X, Y = np.meshgrid(x_vals, y_vals)  # rows=y (T), cols=x (K)

            if metric == "price":
                Z = bs_price_vectorized(S, X, Y, r, sigma, option_type=option_type, q=q)
                cbar = f"{option_type.capitalize()} Price"
                if colorscale is None:
                    cs, zmid_val = "Cividis", None
            else:
                G = bs_greeks_vectorized(S, X, Y, r, sigma, option_type=option_type, q=q)
                Z = G[metric]
                cbar = metric.capitalize()

            fig = go.Figure(data=go.Heatmap(
                z=Z, x=x_vals, y=y_vals,
                colorscale=cs,
                zmid=zmid_val,
                colorbar=dict(title=cbar),
                hovertemplate="K=%{x:.4g}<br>T=%{y:.4f}<br>"+cbar+"=%{z:.6f}<extra></extra>"
            ))
            fig.update_layout(
                title=f"Black-Scholes {cbar} Heatmap — axes: K vs T",
                xaxis_title="Strike (K)",
                yaxis_title="Time to Maturity (T, years)",
                height=500
            )
            return fig

        else:
            raise ValueError("axes must be 'S-sigma' or 'K-T'")



    @staticmethod
    def plot_binomial_price_vs_spot(K, T, r, sigma, N, option_type, q, BinomialOption):

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

        option = BinomialClass(S, K, T, r, sigma, N, option_type, q)
        tree = option.get_tree()
        dot = graphviz.Digraph()

        # Add nodes
        for i, level in enumerate(tree):
            for j, value in enumerate(level):
                node_id = f"{i}_{j}"
                dot.node(node_id, f"{value:.2f}")

        # Add edges
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
    def plot_mc_greek_comparison(strikes: np.ndarray, greeks_dict: dict, greek_name: str, title: str):
        fig = go.Figure()
        for method, values in greeks_dict.items():
            fig.add_trace(go.Scatter(x=strikes, y=values, mode='lines+markers', name=method))
        fig.update_layout(
            title=title,
            xaxis_title="Strike (K)",
            yaxis_title=greek_name,
            legend_title="Method",
            height=500
        )
        return fig


    @staticmethod
    def plot_mc_greek_surface(strikes: np.ndarray,maturities: np.ndarray, greek_surface: np.ndarray, 
                              greek_name: str, fill_nan: float | None = None):

        K = np.asarray(strikes, dtype=float).reshape(-1)
        T = np.asarray(maturities, dtype=float).reshape(-1)
        Z = np.asarray(greek_surface, dtype=float)


        if K.size == 0 or T.size == 0:
            raise ValueError("Empty strikes or maturities.")
        if Z.ndim != 2:
            raise ValueError(f"greek_surface must be 2D, got ndim={Z.ndim}.")


        expected = (T.size, K.size)
        if Z.shape != expected:
            if Z.shape == (K.size, T.size):
                Z = Z.T  
            else:
                raise ValueError(
                    f"greek_surface has shape {Z.shape}, expected {expected} "
                    f"or transposed {(K.size, T.size)}."
                )


        K_sorted_idx = np.argsort(K)
        T_sorted_idx = np.argsort(T)
        if not np.all(K_sorted_idx == np.arange(K.size)):
            K = K[K_sorted_idx]
            Z = Z[:, K_sorted_idx]
        if not np.all(T_sorted_idx == np.arange(T.size)):
            T = T[T_sorted_idx]
            Z = Z[T_sorted_idx, :]

        
        bad = ~np.isfinite(Z)
        if bad.any():
            if fill_nan is None:
                
                Z = Z.copy()
                Z[bad] = 0.0
            else:
                Z = Z.copy()
                Z[bad] = float(fill_nan)

        
        K_mesh, T_mesh = np.meshgrid(K, T, indexing="xy")

        
        fig = go.Figure(data=[go.Surface(x=K_mesh, y=T_mesh, z=Z, name=greek_name)])
        fig.update_layout(
            title=f"{greek_name} Surface via Monte Carlo",
            scene=dict(
                xaxis_title="Strike (K)",
                yaxis_title="Maturity (T)",
                zaxis_title=greek_name
            ),
            height=600
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
        fig.add_trace(go.Scatter(x=k, y=iv_market, mode="markers", name="Market IV", marker=dict(size=8)))
        fig.add_trace(go.Scatter(x=k, y=iv_model, mode="lines+markers", name="SVI Fit", line=dict(dash="solid")))

        fig.update_layout(
            title=f"SVI Volatility Smile (T = {T:.2f} yrs)",
            xaxis_title="Log-Moneyness (k = log(K/F))",
            yaxis_title="Implied Volatility",
            legend_title="Vol Type",
            height=500
        )

        return fig



    @staticmethod
    def plot_svi_vol_surface(k_matrix, maturities, vol_surface):
        K_mesh, T_mesh = np.meshgrid(k_matrix[0], maturities, indexing="ij")

        fig = go.Figure(data=[go.Surface(
            x=K_mesh,
            y=T_mesh,
            z=vol_surface.T,
            colorscale="Viridis"
        )])

        fig.update_layout(
            title="SVI Volatility Surface",
            scene=dict(
                xaxis_title="Log-Moneyness (k)",
                yaxis_title="Maturity (T, years)",
                zaxis_title="Implied Volatility"
            ),
            height=600
        )

        return fig



    @staticmethod
    def plot_sabr_fit_surface(K, market_vols, sabr_vols, F, T):
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
            mode="lines+markers",
            name="SABR Fit",
            line=dict(color="firebrick", width=2, dash="dash")
        ))

        fig.update_layout(
            title=f"SABR Volatility Smile (T = {T:.2f} yrs, F = {F})",
            xaxis_title="Strike Price (K)",
            yaxis_title="Implied Volatility",
            legend_title="Vol Type",
            height=500
        )

        return fig

    
    @staticmethod
    def plot_sabr_vol_surface(strike_matrix, maturity_vector, vol_surface):
        K_mesh, T_mesh = np.meshgrid(strike_matrix[0], maturity_vector, indexing="ij")

        fig = go.Figure(data=[go.Surface(
            x=K_mesh, y=T_mesh, z=vol_surface.T,
            colorscale="Viridis"
        )])

        fig.update_layout(
            title="SABR Volatility Surface",
            scene=dict(
                xaxis_title="Strike (K)",
                yaxis_title="Maturity (T, years)",
                zaxis_title="Implied Volatility"
            ),
            height=600
        )

        return fig


    @staticmethod
    def plot_local_vol_surface(strikes: np.ndarray, maturities: np.ndarray, local_vol_grid: np.ndarray):
        T_mesh, K_mesh = np.meshgrid(maturities, strikes, indexing="ij")  # (len(T), len(K)) para z-grid

        st.markdown("### Preview local volatility grid")
        st.dataframe(local_vol_grid)

        num_total = local_vol_grid.size
        num_valid = np.count_nonzero(~np.isnan(local_vol_grid))
        min_val = np.nanmin(local_vol_grid)
        max_val = np.nanmax(local_vol_grid)

        st.markdown(f"- 🔍 **Min:** {min_val:.4f} &nbsp;&nbsp;&nbsp; **Max:** {max_val:.4f}")
        st.markdown(f"- ✅ **Valores válidos:** {num_valid} / {num_total} ({100 * num_valid / num_total:.1f}%)")

        if np.isnan(local_vol_grid).all():
            st.error("⚠️ Volatility surface is completely NaN.")
            return None

        if num_valid < 0.3 * num_total:
            st.warning("⚠️ Less than 30% of the surface contains valid values. The chart may appear incomplete.")

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
                prices.append(np.nan)  # In case of error, better NaN than break the chart

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
            title="Heston Model: Option Price vs Strike",
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
            title="Heston Calibration: Market vs Model Prices",
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
    def plot_garch_var_bar(var_array: np.ndarray,
                           title: str = "GARCH VaR (1-day ahead forecast)"):
        """
        Plot a single-bar chart for the GARCH VaR value.
        Automatically handles NaNs or empty arrays.
        """
        v = float(np.asarray(var_array).reshape(-1)[-1]) if var_array.size else np.nan
        fig = go.Figure()

        if np.isfinite(v):
            fig.add_trace(go.Bar(x=["VaR (1d)"], y=[abs(v)], name="GARCH VaR (1-day)"))
            fig.update_yaxes(title_text="VaR", rangemode="tozero")
        else:
            fig.add_annotation(text="VaR not available", x=0.5, y=0.5, showarrow=False)
            fig.update_xaxes(visible=False)
            fig.update_yaxes(visible=False)

        fig.update_layout(title=title, height=400)
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
            title="Risk Ratios Overview",
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


    @staticmethod
    def plot_hedging_error_over_time(time_grid: np.ndarray, hedging_errors: np.ndarray, title: str = "Delta Hedging Error Over Time") -> go.Figure:

        hedging_errors = np.atleast_2d(hedging_errors)

        # Mean error over time (across all paths)
        mean_error = np.mean(hedging_errors, axis=0)

        fig = go.Figure()

        fig.add_trace(go.Scatter(
            x=time_grid,
            y=mean_error,
            mode="lines+markers",
            name="Mean Hedging Error",
            line=dict(color="crimson", width=2),
            marker=dict(size=4),
            hovertemplate="Time: %{x:.2f}<br>Error: %{y:.5f}<extra></extra>"
        ))

        fig.add_trace(go.Scatter(
            x=time_grid,
            y=np.zeros_like(time_grid),
            mode="lines",
            name="Zero Error",
            line=dict(color="lightgray", dash="dash"),
            showlegend=True
        ))

        fig.update_layout(
            title=title,
            xaxis_title="Time to Maturity",
            yaxis_title="Hedging Error",
            template="plotly_white",
            height=500,
            width=800,
            margin=dict(l=60, r=60, t=60, b=60),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5)
        )

        return fig
    

    @staticmethod
    def plot_hedging_pnl_decomposition(pnl_dict: dict, time_grid: np.ndarray):
        """
        Stacked bar plot of delta, theta and residual PnL components.
        """
        steps = len(pnl_dict["delta_pnl"])
        time = time_grid[:steps]

        fig = go.Figure()
        fig.add_trace(go.Bar(name="Delta PnL", x=time, y=pnl_dict["delta_pnl"]))
        fig.add_trace(go.Bar(name="Theta PnL", x=time, y=pnl_dict["theta_pnl"]))
        fig.add_trace(go.Bar(name="Residual PnL", x=time, y=pnl_dict["residual_pnl"]))

        fig.update_layout(
            title="PnL Decomposition (Delta + Theta + Residual)",
            barmode='stack',
            xaxis_title="Time",
            yaxis_title="PnL",
            legend_title="Component",
            height=500
        )
        return fig


    @staticmethod
    def plot_total_pnl_cumulative(pnl_dict: dict, time_grid: np.ndarray):
        """
        Line chart of cumulative total PnL over time.
        """
        steps = len(pnl_dict["total_pnl"])
        time = time_grid[:steps]
        cumulative_pnl = np.cumsum(pnl_dict["total_pnl"])

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=time, y=cumulative_pnl, mode="lines+markers", name="Cumulative Total PnL"))

        fig.update_layout(
            title="Cumulative Total PnL Over Time",
            xaxis_title="Time",
            yaxis_title="Cumulative PnL",
            height=500
        )
        return fig
    


    @staticmethod
    def plot_binomial_tree_from_nodes(
        node_tree,
        title="Binomial Tree",
        show_axes=True,
        show_grid=True,
        show_level_labels=True,
        highlight_node=None,       # (i, j) or None
        highlight_path=True,       # emphasize path from root to (i,j)
        show_indices=False         # annotate i,j on selected
    ):
        """
        Visual, orientation-first binomial tree:
        - Grid + integer ticks for level (y).
        - Optional left-side "Level i" labels.
        - Optional highlight of a node (i,j) and its path from the root.
        - Crosshair lines at the selected level and x-position.

        node_tree: list of levels, each with dict nodes {'S','V','Delta','Gamma'}
        """
        N = len(node_tree) - 1

        # Layout positions (centered per level). y = i (0..N). We'll show Level 0 at top.
        node_xy = {(i, j): (j - i/2, i) for i in range(N + 1) for j in range(i + 1)}

        # --- Base edges (light) ---
        xe, ye = [], []
        for i in range(N):
            for j in range(i + 1):
                x0, y0 = node_xy[(i, j)]
                x1, y1 = node_xy[(i + 1, j)]
                x2, y2 = node_xy[(i + 1, j + 1)]
                xe += [x0, x1, None, x0, x2, None]
                ye += [y0, y1, None, y0, y2, None]
        edges_base = go.Scatter(x=xe, y=ye, mode="lines", hoverinfo="skip", opacity=0.35)

        # --- Base nodes (light) ---
        xs, ys, hovers = [], [], []
        for i, level in enumerate(node_tree):
            for j, node in enumerate(level):
                x, y = node_xy[(i, j)]
                xs.append(x); ys.append(y)
                S = node.get("S"); V = node.get("V")
                D = node.get("Delta"); G = node.get("Gamma")
                lines = [f"S={S:.2f}"]
                if V is not None: lines.append(f"V={V:.2f}")
                if D is not None and D == D: lines.append(f"Δ={D:.4f}")
                if G is not None and G == G: lines.append(f"Γ={G:.4f}")
                hovers.append("<br>".join(lines))
        node_size = max(10, 22 - int(N / 2))
        nodes_base = go.Scatter(
            x=xs, y=ys, mode="markers",
            hovertemplate="%{text}<extra></extra>", text=hovers,
            marker=dict(size=node_size, line=dict(width=1)),
            opacity=0.5
        )

        fig = go.Figure([edges_base, nodes_base])

        # --- Highlight (node + path + crosshair) ---
        annotations = []
        shapes = []
        if highlight_node is not None:
            hi, hj = highlight_node
            # Clamp
            hi = max(0, min(N, hi))
            hj = max(0, min(hi, hj))
            hx, hy = node_xy[(hi, hj)]

            # Crosshair lines (level and x-position)
            shapes += [
                dict(type="line", x0=min(xs), x1=max(xs), y0=hy, y1=hy),   # horizontal at level
                dict(type="line", x0=hx, x1=hx, y0=0, y1=N)                # vertical through node
            ]

            # Emphasize path from root to (hi,hj)
            if highlight_path and hi > 0:
                px, py = [], []
                ii, jj = 0, 0
                px.append(node_xy[(0, 0)][0]); py.append(node_xy[(0, 0)][1])
                # Always: from (i,j) next step is either (i+1, j) or (i+1, j+1). To reach (hi,hj),
                # perform (hi - hj) "down-left" moves and (hj) "down-right" moves in any order.
                # We draw a monotone path: first go right jj times, then left the rest.
                for _ in range(jj, hj):  # right moves
                    x1, y1 = node_xy[(ii + 1, jj + 1)]
                    px += [x1]; py += [y1]; ii += 1; jj += 1
                for _ in range(ii, hi):  # left moves
                    x1, y1 = node_xy[(ii + 1, jj)]
                    px += [x1]; py += [y1]; ii += 1
                path_trace = go.Scatter(x=px, y=py, mode="lines", hoverinfo="skip")
                fig.add_trace(path_trace)

            # Highlighted node (bigger, full opacity)
            node_hi = go.Scatter(
                x=[hx], y=[hy], mode="markers",
                hovertemplate="%{text}<extra></extra>",
                text=[hovers[sum(range(hi+1)) + hj] if hovers else ""],
                marker=dict(size=node_size + 8, line=dict(width=2))
            )
            fig.add_trace(node_hi)

            # Optional badge with indices
            if show_indices:
                annotations.append(dict(
                    x=hx, y=max(0, hy - 0.7),
                    text=f"(i={hi}, j={hj})",
                    showarrow=False, xanchor="center", yanchor="top"
                ))

        # Axes & grid
        fig.update_xaxes(visible=show_axes, showgrid=show_grid, dtick=1)
        fig.update_yaxes(visible=show_axes, showgrid=show_grid, dtick=1,
                         autorange="reversed", title_text="Level" if show_axes else None)

        # Level labels at left
        if show_level_labels:
            min_x = (min(xs) if xs else -N/2) - 0.8
            for i in range(N + 1):
                annotations.append(dict(
                    x=min_x, y=i, text=f"Level {i}",
                    showarrow=False, xanchor="right", yanchor="middle", align="right"
                ))
            fig.update_layout(margin=dict(l=80, r=10, t=40, b=10))
        else:
            fig.update_layout(margin=dict(l=10, r=10, t=40, b=10))

        if shapes:
            fig.update_layout(shapes=shapes)
        if annotations:
            fig.update_layout(annotations=annotations)

        fig.update_layout(title=title, showlegend=False)
        return fig