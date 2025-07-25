import streamlit as st
from streamlit_option_menu import option_menu
import numpy as np

# Añade imports de tus módulos
from src.models.pricing_black_scholes import BlackScholesOption
from src.utils.plot_utils import (
    plot_price_vs_spot,
    plot_greeks_vs_spot,
    plot_implied_volatility_vs_market_price_newton
)

# --- Page config ---
st.set_page_config(
    page_title="Option Pricing Model",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# --- Header ---
st.title("📈 Option Pricing Interface")

# --- Model selector ---
model_selected = option_menu(
    menu_title=None,
    options=["Black-Scholes"],  # Más modelos se agregan aquí luego
    icons=["calculator"],
    orientation="horizontal"
)

# --- Black-Scholes UI ---
if model_selected == "Black-Scholes":
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

        submit = st.form_submit_button("Calculate")

    if submit:
        # Validate inputs
        if S <= 0 or K <= 0 or T <= 0 or sigma <= 0:
            st.error("Spot, Strike, Time to Maturity and Volatility must be positive and greater than zero.")
        else:
            # Calculate implied volatility if requested
            if use_iv and market_price is not None and market_price > 0:
                try:
                    implied_vol = BlackScholesOption.implied_volatility_newton(
                        market_price=market_price,
                        S=S,
                        K=K,
                        T=T,
                        r=r,
                        option_type=option_type,
                        q=q
                    )
                    st.success(f"Implied Volatility: {implied_vol:.4%}")
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
            fig_price = plot_price_vs_spot(K, T, r, sigma_used, option_type, q, BlackScholesOption)
            fig_greeks = plot_greeks_vs_spot(K, T, r, sigma_used, option_type, q, BlackScholesOption)

            st.plotly_chart(fig_price, use_container_width=True)
            st.plotly_chart(fig_greeks, use_container_width=True)

            # Implied Volatility Plot if applicable
            if use_iv and market_price is not None and market_price > 0:
                market_prices_array = np.linspace(market_price * 0.5, market_price * 1.5, 50)
                fig_iv = plot_implied_volatility_vs_market_price_newton(
                    S, K, T, r, market_prices_array, option_type, q
                )
                st.plotly_chart(fig_iv, use_container_width=True)
