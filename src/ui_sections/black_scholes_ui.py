import streamlit as st
import numpy as np

from src.models.pricing_black_scholes import BlackScholesOption
from src.models.implied_volatility import ImpliedVolatility
from src.utils.plot_utils import PlotUtils


def black_scholes_ui():
    st.markdown("### 📈 Black-Scholes Option Pricing")

    with st.form("black_scholes_form"):
        st.markdown("#### Option Parameters")
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

        st.markdown("#### Heatmap Settings")
        col3, col4 = st.columns(2)
        with col3:
            S_min = st.number_input("Min Spot Price", value=50.0, min_value=1.0)
            S_max = st.number_input("Max Spot Price", value=150.0, min_value=S_min + 1)
        with col4:
            sigma_min = st.number_input("Min Volatility", value=0.05, min_value=0.001)
            sigma_max = st.number_input("Max Volatility", value=0.5, min_value=sigma_min + 0.01)
        resolution = st.slider("Heatmap Resolution", 10, 100, 50)

        submitted = st.form_submit_button("Calculate")

    if submitted:
        if S <= 0 or K <= 0 or T <= 0 or sigma <= 0:
            st.error("Inputs must be strictly positive.")
            return

        sigma_used = sigma
        if use_iv and market_price:
            try:
                if iv_method == "newton":
                    implied_vol = ImpliedVolatility.implied_volatility_newton(
                        market_price, S, K, T, r, option_type, q
                    )
                elif iv_method == "bisection":
                    implied_vol = ImpliedVolatility.implied_volatility_bisection(
                        market_price, S, K, T, r, option_type, q
                    )
                else:
                    implied_vol_arr = ImpliedVolatility.implied_volatility_vectorized(
                        np.array([market_price]), S, K, T, r, option_type, q, method="newton"
                    )
                    implied_vol = implied_vol_arr[0]
                sigma_used = implied_vol
                st.success(f"Implied Volatility ({iv_method}): {implied_vol:.4%}")
            except RuntimeError as e:
                st.error(f"IV Calculation Error: {e}")

        option = BlackScholesOption(S, K, T, r, sigma_used, option_type, q)
        price = option.price()
        greeks = option.greeks()

        st.success(f"Option Price: {price:.4f}")
        st.markdown("#### Option Greeks")
        col_a, col_b, col_c = st.columns(3)
        col_a.metric("Delta", f"{greeks['delta']:.4f}")
        col_b.metric("Gamma", f"{greeks['gamma']:.4f}")
        col_c.metric("Vega", f"{greeks['vega']:.4f}")
        col_a.metric("Theta", f"{greeks['theta']:.4f}")
        col_b.metric("Rho", f"{greeks['rho']:.4f}")

        st.divider()
        st.markdown("#### 🔍 Price and Greeks vs Spot")
        fig_price = PlotUtils.plot_price_vs_spot(K, T, r, sigma_used, option_type, q, BlackScholesOption)
        fig_greeks = PlotUtils.plot_greeks_vs_spot(K, T, r, sigma_used, option_type, q, BlackScholesOption)
        st.plotly_chart(fig_price, use_container_width=True)
        st.plotly_chart(fig_greeks, use_container_width=True)

        if use_iv and market_price:
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
            st.markdown("#### Implied Volatility vs Market Price")
            st.plotly_chart(fig_iv, use_container_width=True)

        st.divider()
        st.markdown("#### 🔥 Heatmaps")
        fig_heatmap_call, fig_heatmap_put = PlotUtils.plot_black_scholes_heatmaps(
            K, T, r, q, S_min, S_max, sigma_min, sigma_max, resolution
        )
        st.plotly_chart(fig_heatmap_call, use_container_width=True)
        st.plotly_chart(fig_heatmap_put, use_container_width=True)
