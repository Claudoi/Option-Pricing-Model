import streamlit as st
import numpy as np

from src.models.pricing_black_scholes import BlackScholesOption
from src.models.implied_volatility import ImpliedVolatility
from src.utils.plot_utils import PlotUtils


def black_scholes_ui():
    # Header
    st.markdown("## Black-Scholes Option Pricing")
    st.markdown('<div class="small-muted">Analytical pricing • Greeks • IV</div>', unsafe_allow_html=True)

    # Input Form
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

        cHM1, cHM2, cHM3 = st.columns(3)
        with cHM1:
            heatmap_metric = st.selectbox(
                "Metric",
                ["price", "delta", "gamma", "vega", "theta", "rho"],
                help="Choose what the heatmap colors represent."
            )
        with cHM2:
            heatmap_axes = st.selectbox(
                "Axes",
                ["S–σ", "K–T"],
                help="Choose the two variables for the heatmap plane."
            )
        with cHM3:
            resolution = st.slider("Resolution", min_value=10, max_value=150, value=50,
                                   help="Number of grid steps for the heatmap.")

        # Dynamic ranges based on axes
        if heatmap_axes == "S–σ":
            c3, c4 = st.columns(2)
            with c3:
                S_min = st.number_input("Min Spot Price", value=50.0, min_value=1.0)
                S_max = st.number_input("Max Spot Price", value=150.0, min_value=S_min + 1.0)
            with c4:
                sigma_min = st.number_input("Min Volatility", value=0.05, min_value=0.001)
                sigma_max = st.number_input("Max Volatility", value=0.50, min_value=sigma_min + 0.01)

            center_on_iv = False
            if use_iv:
                center_on_iv = st.checkbox("Center σ range around implied vol (if computed)", value=False)
        else:
            # K–T axes ranges
            c5, c6 = st.columns(2)
            with c5:
                K_min = st.number_input("Min Strike (K)", value=max(1e-3, 0.5 * K))
                K_max = st.number_input("Max Strike (K)", value=max(K_min + 1e-3, 1.5 * K))
            with c6:
                T_min = st.number_input("Min Time (years)", value=max(1e-4, 0.1 * T), format="%.4f")
                T_max = st.number_input("Max Time (years)", value=max(T_min + 1e-4, 2.0 * T), format="%.4f")

        submitted = st.form_submit_button("Calculate")
    st.markdown('</div>', unsafe_allow_html=True)  # close card

    # Logic
    if not submitted:
        return

    # Basic input validations
    if any(v <= 0 for v in [S, K, T]) or sigma <= 0:
        st.error("Inputs must be strictly positive (S, K, T, σ).")
        return

    # IV computation (optional)
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

    # Optionally recenter sigma range around IV
    if heatmap_axes == "S–σ":
        if use_iv and market_price and 'center_on_iv' in locals() and center_on_iv:
            span = max(0.05, sigma_used * 0.75)
            sigma_min = max(1e-4, sigma_used - span / 2)
            sigma_max = sigma_used + span / 2

        # Validate ranges
        if S_min >= S_max:
            st.error("Heatmap: Min Spot Price must be < Max Spot Price.")
            return
        if sigma_min >= sigma_max:
            st.error("Heatmap: Min Volatility must be < Max Volatility.")
            return
    else:
        if K_min >= K_max:
            st.error("Heatmap: Min Strike must be < Max Strike.")
            return
        if T_min >= T_max:
            st.error("Heatmap: Min Time must be < Max Time.")
            return

    # Point computation (price + greeks)
    option = BlackScholesOption(S, K, T, r, sigma_used, option_type, q)
    price = option.price()
    greeks = option.greeks()

    # Results Card
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

    # Visualizations
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

    # Heatmaps
    st.markdown('<div class="card" style="padding:1rem;">', unsafe_allow_html=True)
    st.markdown("#### Heatmaps")

    # 1) both heatmaps (call/put price over S–σ)
    if heatmap_axes == "S–σ" and heatmap_metric == "price":
        fig_heatmap_call, fig_heatmap_put = PlotUtils.plot_black_scholes_heatmaps(
            K, T, r, q, S_min, S_max, sigma_min, sigma_max, resolution
        )
        st.plotly_chart(fig_heatmap_call, use_container_width=True)
        st.plotly_chart(fig_heatmap_put, use_container_width=True)

    # 2) Heatmap flexible (metrics and axes configurables)
    @st.cache_data(show_spinner=False)
    def _flex_heatmap(axes, metric, option_type, S, K, T, r, q, sigma,
                      S_min, S_max, sigma_min, sigma_max, K_min, K_max, T_min, T_max, resolution):
        return PlotUtils.plot_bs_heatmap_flexible(
            axes=("S-sigma" if axes == "S–σ" else "K-T"),
            metric=metric,
            option_type=option_type,
            S=S, K=K, T=T, r=r, q=q, sigma=sigma,
            S_min=S_min, S_max=S_max, sigma_min=sigma_min, sigma_max=sigma_max,
            K_min=K_min, K_max=K_max, T_min=T_min, T_max=T_max,
            resolution=resolution
        )

    if heatmap_axes == "S–σ":
        fig_flex = _flex_heatmap(heatmap_axes, heatmap_metric, option_type, S, K, T, r, q, sigma_used,
                                 S_min, S_max, sigma_min, sigma_max, None, None, None, None, resolution)
    else:
        fig_flex = _flex_heatmap(heatmap_axes, heatmap_metric, option_type, S, K, T, r, q, sigma_used,
                                 None, None, None, None, K_min, K_max, T_min, T_max, resolution)

    st.plotly_chart(fig_flex, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)
