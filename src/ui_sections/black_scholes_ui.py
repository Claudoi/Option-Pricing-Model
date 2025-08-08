import streamlit as st
import numpy as np

from src.models.pricing_black_scholes import BlackScholesOption
from src.models.implied_volatility import ImpliedVolatility
from src.utils.plot_utils import PlotUtils


def black_scholes_ui():
    # ------- SECTION HEADER -------
    st.markdown("## Black-Scholes Option Pricing")
    st.markdown('<div class="small-muted">Analytical pricing • Greeks • IV</div>', unsafe_allow_html=True)

    # ------- INPUT FORM INSIDE A “CARD” -------
    st.markdown('<div class="card" style="padding:1rem;">', unsafe_allow_html=True)
    with st.form("black_scholes_form"):
        st.markdown("#### Option Parameters")

        # Two-column layout for parameters
        c1, c2 = st.columns(2)
        with c1:
            S = st.number_input("Spot Price (S)", value=100.0, min_value=0.01, format="%.2f",
                                help="Current underlying price.")
            K = st.number_input("Strike Price (K)", value=100.0, min_value=0.01, format="%.2f",
                                help="Strike of the option.")
            T = st.number_input("Time to Maturity (T, years)", value=1.0, min_value=1e-4, format="%.4f",
                                help="Expressed in years (e.g., 0.5 = 6 months).")
            q = st.number_input("Dividend Yield (q)", value=0.0, min_value=0.0, format="%.4f")

        with c2:
            r = st.number_input("Risk-Free Rate (r)", value=0.05, min_value=0.0, format="%.4f")
            sigma = st.number_input("Volatility (σ)", value=0.20, min_value=1e-4, format="%.4f")
            option_type = st.selectbox("Option Type", ["call", "put"])
            use_iv = st.checkbox("Calculate Implied Volatility from Market Price", value=False)
            market_price, iv_method = None, None
            if use_iv:
                market_price = st.number_input("Market Option Price", min_value=0.01, format="%.4f")
                iv_method = st.selectbox("Implied Volatility Method", ["newton", "bisection", "vectorized"])

        # Heatmap configuration
        st.markdown("---")
        st.markdown("#### Heatmap Settings")
        c3, c4 = st.columns(2)
        with c3:
            S_min = st.number_input("Min Spot Price", value=50.0, min_value=1.0)
            S_max = st.number_input("Max Spot Price", value=150.0, min_value=S_min + 1.0)
        with c4:
            sigma_min = st.number_input("Min Volatility", value=0.05, min_value=0.001)
            sigma_max = st.number_input("Max Volatility", value=0.50, min_value=sigma_min + 0.01)

        resolution = st.slider("Heatmap Resolution", min_value=10, max_value=100, value=50,
                               help="Number of grid steps for the heatmap.")

        submitted = st.form_submit_button("Calculate")
    st.markdown('</div>', unsafe_allow_html=True)  # close card

    # ------- CALCULATION LOGIC -------
    if not submitted:
        return

    # Basic input validations
    if any(v <= 0 for v in [S, K, T]) or sigma <= 0:
        st.error("Inputs must be strictly positive (S, K, T, σ).")
        return
    if S_min >= S_max:
        st.error("Heatmap: Min Spot Price must be < Max Spot Price.")
        return
    if sigma_min >= sigma_max:
        st.error("Heatmap: Min Volatility must be < Max Volatility.")
        return

    # If IV calculation is enabled, compute implied volatility and replace sigma
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
            else:  # vectorized for a single point
                implied_vol_arr = ImpliedVolatility.implied_volatility_vectorized(
                    np.array([market_price]), S, K, T, r, option_type, q, method="newton"
                )
                implied_vol = float(implied_vol_arr[0])
            sigma_used = implied_vol
            st.success(f"Implied Volatility ({iv_method}): {implied_vol:.4%}")
        except Exception as e:
            st.error(f"IV Calculation Error: {e}")
            return

    # Calculate price and Greeks
    option = BlackScholesOption(S, K, T, r, sigma_used, option_type, q)
    price = option.price()
    greeks = option.greeks()

    # ------- RESULTS CARD -------
    st.markdown('<div class="card" style="padding:1rem;">', unsafe_allow_html=True)
    st.markdown("#### Results")
    cA, cB, cC, cD = st.columns(4)
    cA.metric("Price", f"{price:.4f}")
    cB.metric("Delta", f"{greeks['delta']:.4f}")
    cC.metric("Gamma", f"{greeks['gamma']:.4f}")
    cD.metric("Vega", f"{greeks['vega']:.4f}")

    cE, cF, _ = st.columns(3)
    cE.metric("Theta", f"{greeks['theta']:.4f}")
    cF.metric("Rho", f"{greeks['rho']:.4f}")
    st.markdown('</div>', unsafe_allow_html=True)

    # ------- VISUALIZATIONS IN TABS -------
    st.markdown('<div class="card" style="padding:1rem;">', unsafe_allow_html=True)
    st.markdown("#### Visualizations")
    t1, t2, t3 = st.tabs(["Price vs Spot", "Greeks vs Spot", "Implied Volatility"])

    with t1:
        fig_price = PlotUtils.plot_price_vs_spot(K, T, r, sigma_used, option_type, q, BlackScholesOption)
        st.plotly_chart(fig_price, use_container_width=True)

    with t2:
        fig_greeks = PlotUtils.plot_greeks_vs_spot(K, T, r, sigma_used, option_type, q, BlackScholesOption)
        st.plotly_chart(fig_greeks, use_container_width=True)

    with t3:
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
            st.plotly_chart(fig_iv, use_container_width=True)
        else:
            st.info("Enable “Calculate Implied Volatility” to see this chart.")
    st.markdown('</div>', unsafe_allow_html=True)

    # ------- HEATMAPS CARD -------
    st.markdown('<div class="card" style="padding:1rem;">', unsafe_allow_html=True)
    st.markdown("#### 🔥 Heatmaps")
    fig_heatmap_call, fig_heatmap_put = PlotUtils.plot_black_scholes_heatmaps(
        K, T, r, q, S_min, S_max, sigma_min, sigma_max, resolution
    )
    st.plotly_chart(fig_heatmap_call, use_container_width=True)
    st.plotly_chart(fig_heatmap_put, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)
