import streamlit as st
from streamlit_option_menu import option_menu
import numpy as np
import plotly.graph_objs as go
import pandas as pd

import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.models.pricing_black_scholes import BlackScholesOption
from src.models.pricing_binomial import BinomialOption
from src.models.pricing_montecarlo import MonteCarloOption
from src.models.greeks import BlackScholesGreeks
from src.risk.risk_analysis import PortfolioVaR, HistoricalVaR, MonteCarloVaR
from src.risk.risk_rolling import RollingVaR
from src.risk.risk_ratios import RiskRatios
from src.models.implied_volatility import ImpliedVolatility
from src.volatility.local_volatility import LocalVolatilitySurface
from src.volatility.sabr_calibration import SABRCalibrator
from src.volatility.stochastic_volatility import HestonModel
from src.volatility.svi_calibration import SVI_Calibrator
from src.volatility.volatility_surface import VolatilitySurface


from src.utils.plot_utils import PlotUtils
from src.utils.utils import fetch_returns_from_yahoo


# --- Page config ---
st.set_page_config(
    page_title="Option Pricing Model",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# --- Header ---
col1, col2 = st.columns([0.9, 0.1])
with col1:
    st.markdown("## 💻 Option Pricing Interface")

# --- Option menu ---
selected = option_menu(
    menu_title=None,
    options=["Black-Scholes", "Binomial", "Monte Carlo", "Risk Analysis", "Volatility"],
    icons=["calculator", "tree", "shuffle", "activity", "bar-chart"],
    orientation="horizontal"
)





# --- Black-Scholes Section ---
if selected == "Black-Scholes":
    st.header("Black-Scholes Option Pricing")

    with st.form("bs_form"):
        col1, col2 = st.columns(2)

        with col1:
            S = st.number_input("Spot Price (S)", value=100.0, min_value=0.01, format="%.2f")
            K = st.number_input("Strike Price (K)", value=100.0, min_value=0.01, format="%.2f")
            T = st.number_input("Time to Maturity (T in years)", value=1.0, min_value=0.0001, format="%.4f")
            q = st.number_input("Dividend Yield (q)", value=0.0, min_value=0.0, format="%.4f")

        with col2:
            r = st.number_input("Risk-Free Rate (r)", value=0.05, min_value=0.0, format="%.4f")
            sigma = st.number_input("Volatility (σ)", value=0.2, min_value=0.0001, format="%.4f")
            option_type = st.selectbox("Option Type", ["call", "put"])
            use_iv = st.checkbox("Calculate Implied Volatility from Market Price")
            market_price = None
            if use_iv:
                market_price = st.number_input("Market Option Price", min_value=0.01, format="%.4f")
                iv_method = st.selectbox("Implied Volatility Method", ["newton", "bisection", "vectorized"])

        # Heatmap input parameters
        st.markdown("---")
        st.subheader("Black-Scholes Price Heatmap Settings")
        S_min = st.number_input("Min Spot Price for Heatmap", value=50.0, min_value=1.0)
        S_max = st.number_input("Max Spot Price for Heatmap", value=150.0, min_value=S_min+1)
        sigma_min = st.number_input("Min Volatility for Heatmap", value=0.05, min_value=0.001)
        sigma_max = st.number_input("Max Volatility for Heatmap", value=0.5, min_value=sigma_min+0.01)
        resolution = st.slider("Heatmap Resolution", 10, 100, 50)

        submit = st.form_submit_button("Calculate")

    if submit:
        if S <= 0 or K <= 0 or T <= 0 or sigma <= 0:
            st.error("Spot, Strike, Time to Maturity and Volatility must be positive and greater than zero.")
        else:
            # Calculate implied volatility if requested
            if use_iv and market_price is not None and market_price > 0:
                try:
                    if iv_method == "newton":
                        implied_vol = ImpliedVolatility.implied_volatility_newton(
                            market_price=market_price, S=S, K=K, T=T, r=r,
                            option_type=option_type, q=q
                        )
                    elif iv_method == "bisection":
                        implied_vol = ImpliedVolatility.implied_volatility_bisection(
                            market_price=market_price, S=S, K=K, T=T, r=r,
                            option_type=option_type, q=q
                        )
                    elif iv_method == "vectorized":
                        implied_vol_arr = ImpliedVolatility.implied_volatility_vectorized(
                            np.array([market_price]), S, K, T, r, option_type, q,
                            method="newton"
                        )
                        implied_vol = implied_vol_arr[0]
                    else:
                        implied_vol = sigma  # fallback

                    st.success(f"Implied Volatility ({iv_method}): {implied_vol:.4%}")
                    sigma_used = implied_vol
                except RuntimeError as e:
                    st.error(f"Implied volatility calculation failed: {e}")
                    sigma_used = sigma
            else:
                sigma_used = sigma

            # Price and Greeks
            option = BlackScholesOption(S, K, T, r, sigma_used, option_type, q)
            price = option.price()
            greeks = option.greeks()

            st.success(f"Option Price: {price:.4f}")
            st.subheader("Greeks")
            col_a, col_b, col_c = st.columns(3)
            col_a.metric("Delta", f"{greeks['delta']:.4f}")
            col_b.metric("Gamma", f"{greeks['gamma']:.4f}")
            col_c.metric("Vega", f"{greeks['vega']:.4f}")
            col_a.metric("Theta", f"{greeks['theta']:.4f}")
            col_b.metric("Rho", f"{greeks['rho']:.4f}")

            # Plots
            fig_price = PlotUtils.plot_price_vs_spot(K, T, r, sigma_used, option_type, q, BlackScholesOption)
            fig_greeks = PlotUtils.plot_greeks_vs_spot(K, T, r, sigma_used, option_type, q, BlackScholesOption)
            fig_heatmap_call, fig_heatmap_put = PlotUtils.plot_black_scholes_heatmaps(
                K, T, r, q, S_min, S_max, sigma_min, sigma_max, resolution
            )

            st.plotly_chart(fig_price, use_container_width=True)
            st.plotly_chart(fig_greeks, use_container_width=True)

            if use_iv and market_price is not None and market_price > 0:
                market_prices_array = np.linspace(market_price * 0.5, market_price * 1.5, 50)
                if iv_method == "newton":
                    fig_iv = PlotUtils.plot_implied_volatility_vs_market_price_newton(
                        S, K, T, r, market_prices_array, option_type, q
                    )
                elif iv_method == "bisection":
                    fig_iv = PlotUtils.plot_implied_volatility_vs_market_price_bisection(
                        S, K, T, r, market_prices_array, option_type, q
                    )
                else:
                    fig_iv = PlotUtils.plot_implied_volatility_vs_market_price_vectorized(
                        S, K, T, r, market_prices_array, option_type, q, method="newton"
                    )
                st.plotly_chart(fig_iv, use_container_width=True)

            st.markdown("---")
            st.subheader("Black-Scholes Price Heatmaps")
            st.plotly_chart(fig_heatmap_call, use_container_width=True)
            st.plotly_chart(fig_heatmap_put, use_container_width=True)





# --- Binomial Section ---
if selected == "Binomial":
    st.markdown("### Binomial Option Pricing")

    with st.form("binomial_form"):
        col1, col2 = st.columns(2)
        with col1:
            S = st.number_input("Spot Price (S)", value=100.0, min_value=0.01, format="%.2f")
            K = st.number_input("Strike Price (K)", value=100.0, min_value=0.01, format="%.2f")
            T = st.number_input("Time to Maturity (T in years)", value=1.0, min_value=0.0001, format="%.4f")
            q = st.number_input("Dividend Yield (q)", value=0.0, min_value=0.0, format="%.4f")
        with col2:
            r = st.number_input("Risk-Free Rate (r)", value=0.05, min_value=0.0, format="%.4f")
            sigma = st.number_input("Volatility (σ)", value=0.2, min_value=0.0001, format="%.4f")
            option_type = st.selectbox("Option Type", ["call", "put"])
            N = st.slider("Number of Steps (N)", min_value=1, max_value=100, value=5)
            style = st.selectbox("Option Style", ["European", "American"])

        submitted = st.form_submit_button("Calculate Binomial Price")

    if submitted:
        try:
            bin_opt = BinomialOption(S, K, T, r, sigma, N, option_type, q)
            if style == "European":
                price = bin_opt.price_european()
            else:
                price = bin_opt.price_american()
            st.success(f"Binomial {style} Option Price: {price:.4f}")

            # --- Interactive Plot: Price vs Spot ---
            fig = PlotUtils.plot_binomial_price_vs_spot(
                K, T, r, sigma, N, option_type, q, BinomialOption
            )
            st.plotly_chart(fig, use_container_width=True)

            # --- Show Binomial Tree for small N (N <= 6) ---
            if N <= 6:
                st.markdown("#### Binomial Tree Visualization")
                PlotUtils.show_binomial_tree(S, K, T, r, sigma, N, option_type, q, BinomialOption)

                # Mostrar SIEMPRE el árbol de sensibilidades
                try:
                    st.markdown("#### Sensitividades Locales (Delta/Gamma por nodo)")
                    tree = bin_opt.get_sensitivities_tree(american=(style == "American"))
                    dot = PlotUtils.graphviz_binomial_sensitivities(tree)
                    st.graphviz_chart(dot.source)   # <---- ¡Esto es lo importante!
                except Exception as sensi_err:
                    st.error(f"Error mostrando sensibilidades: {sensi_err}")

        except Exception as e:
            st.error(f"Error in Binomial pricing: {e}")





# --- Monte Carlo Section ---
if selected == "Monte Carlo":
    st.markdown("### Monte Carlo Option Pricing")

    with st.form("mc_form"):
        col1, col2 = st.columns(2)
        with col1:
            S = st.number_input("Spot Price (S)", value=100.0, min_value=0.01, format="%.2f")
            K = st.number_input("Strike Price (K)", value=100.0, min_value=0.01, format="%.2f")
            T = st.number_input("Time to Maturity (T in years)", value=1.0, min_value=0.0001, format="%.4f")
            q = st.number_input("Dividend Yield (q)", value=0.0, min_value=0.0, format="%.4f")
        with col2:
            r = st.number_input("Risk-Free Rate (r)", value=0.05, min_value=0.0, format="%.4f")
            sigma = st.number_input("Volatility (σ)", value=0.2, min_value=0.0001, format="%.4f")
            option_type = st.selectbox("Option Type", ["call", "put"])
            n_sim = st.number_input("Simulations", value=10000, min_value=1000, step=1000)
            n_steps = st.slider("Time Steps", min_value=10, max_value=500, value=100, step=10)
            exotic = st.selectbox(
                "Option Variant",
                ["Vanilla", "Asian (arithmetic)", "Asian (geometric)", "Lookback (fixed)", "Lookback (floating)", "Digital Barrier", "American (Longstaff-Schwartz)"]
            )

        submitted = st.form_submit_button("Calculate Monte Carlo Price")

    if submitted:
        try:
            mc = MonteCarloOption(S, K, T, r, sigma, option_type, n_sim, n_steps, q)

            # --- Precio según la variante seleccionada ---
            if exotic == "Vanilla":
                price = mc.price_vanilla()
            elif exotic == "Asian (arithmetic)":
                price = mc.price_asian()
            elif exotic == "Asian (geometric)":
                price = mc.price_asian_geometric()
            elif exotic == "Lookback (fixed)":
                price = mc.price_lookback(strike_type="fixed")
            elif exotic == "Lookback (floating)":
                price = mc.price_lookback(strike_type="floating")
            elif exotic == "Digital Barrier":
                price = mc.price_digital_barrier(barrier=K * 1.1, barrier_type="up-and-in")
            elif exotic == "American (Longstaff-Schwartz)":
                price = mc.price_american_lsm()
            else:
                price = None

            st.success(f"Monte Carlo {exotic} Price: {price:.4f}")

            # --- Visualizar paths simulados ---
            st.markdown("#### Monte Carlo Simulated Price Paths")
            paths = mc._simulate_paths()
            fig_paths = PlotUtils.plot_mc_paths(paths)
            st.plotly_chart(fig_paths, use_container_width=True)

            # --- Visualizar histograma de payoffs finales ---
            st.markdown("#### Distribución de Payoffs Finales")
            ST = paths[:, -1]
            payoffs = np.maximum(ST - K, 0) if option_type == "call" else np.maximum(K - ST, 0)
            fig_payoff = go.Figure(data=[go.Histogram(x=payoffs, nbinsx=50)])
            fig_payoff.update_layout(
                title="Distribución de Payoffs (Monte Carlo)",
                xaxis_title="Payoff",
                yaxis_title="Frecuencia"
            )
            st.plotly_chart(fig_payoff, use_container_width=True)

            # --- (Opcional) Visualizar griegas estimadas por diferencias finitas ---
            st.markdown("#### Greeks (Estimadas vía Monte Carlo)")
            col1, col2, col3 = st.columns(3)
            delta = mc.greek("delta")
            vega = mc.greek("vega")
            theta = mc.greek("theta")
            rho = mc.greek("rho")
            col1.metric("Delta", f"{delta:.4f}")
            col2.metric("Vega", f"{vega:.4f}")
            col3.metric("Theta", f"{theta:.4f}")
            col1.metric("Rho", f"{rho:.4f}")

        except Exception as e:
            st.error(f"Error in Monte Carlo pricing: {e}")






if selected == "Risk Analysis":
    st.markdown("### Portfolio Risk Analysis & VaR/ES")

    with st.form("risk_data_form"):
        tickers_str = st.text_input("Portfolio Tickers (comma separated)", "AAPL, MSFT")
        tickers = [t.strip().upper() for t in tickers_str.split(",") if t.strip()]
        col1, col2 = st.columns(2)
        with col1:
            start = st.date_input("Start Date", value=pd.to_datetime("2021-01-01"))
        with col2:
            end = st.date_input("End Date", value=pd.to_datetime("today"))
        weights_str = st.text_input("Portfolio Weights (comma separated, must sum 1)", "0.5, 0.5")
        weights = np.array([float(w) for w in weights_str.split(",")])
        method = st.selectbox("Select VaR Method", ["Parametric", "Historical", "Monte Carlo", "Rolling EWMA", "Rolling GARCH"])
        confidence = st.slider("Confidence Level", 0.90, 0.99, 0.95, step=0.01)
        window = st.number_input("Rolling Window (for EWMA/GARCH)", value=100, min_value=10, step=1)
        lambda_ = st.number_input("EWMA Lambda", value=0.94, min_value=0.7, max_value=0.99, step=0.01)
        submitted = st.form_submit_button("Load & Analyze")

    if submitted:
        try:
            # 1. Descargar retornos de Yahoo Finance
            returns = fetch_returns_from_yahoo(tickers, str(start), str(end))
            st.write("#### Sample of returns:", returns.head())
            # 2. Calcular retornos de portafolio
            portfolio_returns = returns @ weights

            # --- Métricas y ratios ---
            risk_ratios = RiskRatios(portfolio_returns)
            ratios_dict = {
                "Sharpe": risk_ratios.sharpe_ratio(),
                "Sortino": risk_ratios.sortino_ratio(),
                "Calmar": risk_ratios.calmar_ratio(),
                "Max Drawdown": risk_ratios.max_drawdown(),
                "Omega": risk_ratios.omega_ratio(),
                "Skewness": risk_ratios.skewness(),
                "Kurtosis": risk_ratios.kurtosis(),
                "VaR 95%": risk_ratios.value_at_risk(0.05),
                "ES 95%": risk_ratios.expected_shortfall(0.05),
            }
            st.markdown("#### Main Risk Ratios")
            cols = st.columns(len(ratios_dict))
            for col, (k, v) in zip(cols, ratios_dict.items()):
                col.metric(
                    label=k,
                    value=f"{v:.5f}" if isinstance(v, float) else v
    )

            # --- VaR / ES ---
            var, es, var_series = None, None, None
            if method == "Parametric":
                var_model = PortfolioVaR(returns, weights, confidence_level=confidence)
                var = var_model.calculate_var()
                es = var_model.calculate_es()
                st.success(f"Parametric VaR: {var:.4%} | ES: {es:.4%}")
            elif method == "Historical":
                var_model = HistoricalVaR(portfolio_returns, confidence_level=confidence)
                var = var_model.calculate_var()
                es = var_model.calculate_es()
                st.success(f"Historical VaR: {var:.4%} | ES: {es:.4%}")
            elif method == "Monte Carlo":
                mu = returns.mean().values
                sigma = returns.std().values
                var_model = MonteCarloVaR(returns.iloc[-1].values, mu, sigma, weights, confidence_level=confidence)
                var, es, simulated_returns = var_model.calculate_var_es()
                st.success(f"Monte Carlo VaR: {var:.4%} | ES: {es:.4%}")
            elif method == "Rolling EWMA":
                rolling = RollingVaR(portfolio_returns, method="ewma", confidence_level=confidence, window=window, lambda_=lambda_)
                var_series = rolling.calculate_var_series()
                st.plotly_chart(
                    PlotUtils.plot_rolling_var(portfolio_returns, var_series, method="ewma", confidence_level=confidence, window=window),
                    use_container_width=True
                )
                st.info("Rolling VaR curve (EWMA shown).")
                var = var_series[-1] if var_series is not None else None
                es = None  # opcional, puedes estimar ES rolling si quieres
            elif method == "Rolling GARCH":
                try:
                    rolling = RollingVaR(portfolio_returns, method="garch", confidence_level=confidence, window=window)
                    var = rolling.calculate_var_series()
                    st.plotly_chart(PlotUtils.plot_garch_var_bar(var), use_container_width=True)
                except ImportError:
                    st.error("You need to install the 'arch' package for GARCH.")

            # --- Histograma de returns con VaR/ES ---
            if var is not None and es is not None:
                st.plotly_chart(
                    PlotUtils.plot_var_es_histogram(
                        portfolio_returns, var, es,
                        title=f"#### Return Distribution with VaR/ES ({int(confidence*100)}%)"
                    ),
                    use_container_width=True
                )

        except Exception as e:
            st.error(f"Error loading data or calculating risk metrics: {e}")







if selected == "Volatility":
    st.markdown("## Volatility Modeling & Calibration")

    vol_tab = st.tabs(["Vol Surface", "SVI Smile", "SABR", "Local Vol", "Heston"])

    # ------------- Volatility Surface --------------
    with vol_tab[0]:
        st.subheader("Volatility Surface (Market Data Interpolation)")
        ticker = st.text_input("Ticker", "AAPL", key="vs_ticker")
        st.write("Carga y visualiza la superficie de volatilidad usando precios de opciones reales.")
        if st.button("Load Vol Surface", key="load_vol_surface_btn"):
            try:
                vol_surface = VolatilitySurface(ticker)
                vol_surface.fetch_data()  # <- IMPORTANTE: cargar datos antes de plot
                fig = PlotUtils.plot_market_vol_surface(vol_surface)
                st.plotly_chart(fig, use_container_width=True)
            except Exception as e:
                st.error(f"Failed to load volatility surface: {e}")

    # ------------- SVI Calibration -----------------
    with vol_tab[1]:
        st.subheader("SVI Smile Calibration")
        k = st.text_area("Log-moneyness (comma separated)", "0, -0.1, 0.1, 0.2", key="svi_k")
        iv = st.text_area("Implied Vols (comma separated)", "0.25, 0.24, 0.26, 0.27", key="svi_iv")
        T_svi = st.number_input("Maturity (years)", value=0.5, format="%.2f", key="svi_maturity")
        if st.button("Calibrate SVI", key="svi_btn"):
            try:
                k_arr = np.array([float(x) for x in k.split(",")])
                iv_arr = np.array([float(x) for x in iv.split(",")])
                svi = SVI_Calibrator(k_arr, iv_arr, T_svi)
                params, iv_fit = svi.calibrate()
                st.success(f"SVI Params: a={params[0]:.4f}, b={params[1]:.4f}, rho={params[2]:.4f}, m={params[3]:.4f}, sigma={params[4]:.4f}")
                fig = PlotUtils.plot_svi_fit(k_arr, iv_arr, iv_fit, T_svi)
                st.plotly_chart(fig, use_container_width=True)
            except Exception as e:
                st.error(f"SVI Calibration failed: {e}")

    # ------------- SABR Calibration ----------------
    with vol_tab[2]:
        st.subheader("SABR Volatility Fit")
        K = st.text_area("Strikes (comma separated)", "90, 100, 110, 120", key="sabr_K")
        market_vols = st.text_area("Market IV (comma separated)", "0.22, 0.21, 0.23, 0.24", key="sabr_market_iv")
        F = st.number_input("Forward Price", value=100.0, key="sabr_forward")
        T_sabr = st.number_input("Maturity (years)", value=0.5, format="%.2f", key="sabr_maturity")
        if st.button("Calibrate SABR", key="sabr_btn"):
            try:
                K_arr = np.array([float(x) for x in K.split(",")])
                market_vols_arr = np.array([float(x) for x in market_vols.split(",")])
                F_val = float(F)
                T_val = float(T_sabr)
                sabr = SABRCalibrator(F_val, K_arr, T_val, market_vols_arr, beta_fixed=0.5)
                params = sabr.calibrate()      # <--- ¡Así!
                sabr_vols = sabr.model_vols()
                st.success(f"SABR Params: alpha={params[0]:.4f}, beta={params[1]:.4f}, rho={params[2]:.4f}, nu={params[3]:.4f}")
                fig = PlotUtils.plot_sabr_fit_surface(K_arr, market_vols_arr, sabr_vols, F_val, T_val)
                st.plotly_chart(fig, use_container_width=True)
            except Exception as e:
                st.error(f"SABR calibration failed: {e}")


    # ------------- Local Volatility ----------------
    with vol_tab[3]:
        st.subheader("Local Volatility Surface (Dupire)")
        strikes = st.text_area("Strikes (comma separated)", "90,100,110,120", key="lv_strikes")
        maturities = st.text_area("Maturities (comma separated, years)", "0.25,0.5,1", key="lv_maturities")
        local_vol_grid = st.file_uploader("Local vol grid (CSV)", type=["csv"], key="lv_csv")
        if st.button("Show Local Vol Surface", key="lv_btn"):
            try:
                strikes_arr = np.array([float(x) for x in strikes.split(",")])
                maturities_arr = np.array([float(x) for x in maturities.split(",")])
                if local_vol_grid:
                    import pandas as pd
                    grid = pd.read_csv(local_vol_grid, header=None).values
                else:
                    # Puedes llamar a tu clase LocalVolatilitySurface aquí si lo tienes implementado.
                    grid = np.ones((len(maturities_arr), len(strikes_arr)))
                fig = PlotUtils.plot_local_vol_surface(strikes_arr, maturities_arr, grid)
                st.plotly_chart(fig, use_container_width=True)
            except Exception as e:
                st.error(f"Local volatility surface error: {e}")

    # ------------- Heston Model --------------------
    with vol_tab[4]:
        st.subheader("Heston Model: Price vs Strike")
        S0 = st.number_input("Spot Price", value=100.0, key="heston_spot")
        T_heston = st.number_input("Maturity (years)", value=1.0, format="%.2f", key="heston_maturity")
        r = st.number_input("Risk-Free Rate", value=0.01, key="heston_rf")
        kappa = st.number_input("kappa", value=2.0, key="heston_kappa")
        theta = st.number_input("theta (long-run var)", value=0.04, key="heston_theta")
        sigma = st.number_input("sigma (vol of vol)", value=0.5, key="heston_sigma")
        rho = st.number_input("rho (correlation)", value=-0.7, key="heston_rho")
        v0 = st.number_input("v0 (init var)", value=0.04, key="heston_v0")
        option_type = st.selectbox("Option Type", ["call", "put"], key="heston_option_type")
        if st.button("Simulate Heston", key="heston_btn"):
            try:
                fig = PlotUtils.plot_heston_price_vs_strike(S0, T_heston, r, kappa, theta, sigma, rho, v0, option_type)
                st.plotly_chart(fig, use_container_width=True)
            except Exception as e:
                st.error(f"Heston simulation failed: {e}")

    # ------------ Sugerencia extra -----------------
    st.markdown(
        """
        <div style='margin-top:2em; font-size:0.9em; color:gray'>
        <b>¿Ideas para ampliar?</b><br>
        · Añadir visualización de la superficie de volatilidad implícita de mercado con animación para diferentes fechas.<br>
        · Integrar calibración automática para varias smiles (distintos T) a la vez.<br>
        · Añadir exportación a Excel de los parámetros calibrados y de las superficies.<br>
        </div>
        """, unsafe_allow_html=True
    )
