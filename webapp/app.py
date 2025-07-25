import streamlit as st
from streamlit_option_menu import option_menu
import numpy as np
import plotly.graph_objs as go

import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.models.pricing_black_scholes import BlackScholesOption
from src.models.pricing_binomial import BinomialOption
from src.models.pricing_montecarlo import MonteCarloOption
from src.models.greeks import BlackScholesGreeks
from src.models.implied_volatility import ImpliedVolatility

from src.utils.plot_utils import PlotUtils

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
