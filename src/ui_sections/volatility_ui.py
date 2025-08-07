import streamlit as st
import pandas as pd
import numpy as np
from src.volatility.volatility_surface import VolatilitySurface
from src.volatility.svi_calibration import SVI_Calibrator
from src.volatility.sabr_calibration import SABRCalibrator
from src.volatility.local_volatility import LocalVolatilitySurface
from src.volatility.stochastic_volatility import calibrate_heston
from src.utils.plot_utils import PlotUtils


def render_volatility_ui():
    st.markdown("## Volatility Modeling & Calibration")
    vol_tabs = st.tabs(["Vol Surface", "SVI", "SABR", "Local Vol", "Heston"])

    render_vol_surface_tab(vol_tabs[0])
    render_svi_tab(vol_tabs[1])
    render_sabr_tab(vol_tabs[2])
    render_local_vol_tab(vol_tabs[3])
    render_heston_tab(vol_tabs[4])


def render_vol_surface_tab(tab):
    with tab:
        st.subheader("Volatility Surface from Market Data")
        ticker = st.text_input("Ticker Symbol", "AAPL")
        if st.button("Load Volatility Surface"):
            try:
                vs = VolatilitySurface(ticker)
                vs.fetch_data()
                fig = PlotUtils.plot_market_vol_surface(vs)
                st.plotly_chart(fig, use_container_width=True)
            except Exception as e:
                st.error(f"Error loading surface: {e}")


def render_svi_tab(tab):
    with tab:
        st.subheader("SVI Calibration")
        mode = st.radio("Mode", ["Single Maturity", "Full Surface"])

        if mode == "Single Maturity":
            k_input = st.text_area("Log-Moneyness", "0, -0.1, 0.1, 0.2")
            iv_input = st.text_area("Market IVs", "0.25, 0.24, 0.26, 0.27")
            T = st.number_input("Maturity (T)", 0.5)

            if st.button("Calibrate SVI"):
                try:
                    k = np.array([float(x) for x in k_input.split(",")])
                    iv = np.array([float(x) for x in iv_input.split(",")])
                    if len(k) != len(iv):
                        st.warning("⚠️ Length mismatch.")
                        return

                    svi = SVI_Calibrator(k, iv, T)
                    params, fitted = svi.calibrate()
                    st.success("Calibrated SVI Parameters:")
                    for name, val in zip(["a", "b", "rho", "m", "sigma"], params):
                        st.write(f"**{name}**: {val:.4f}")
                    fig = PlotUtils.plot_svi_fit(k, iv, fitted, T)
                    st.plotly_chart(fig, use_container_width=True)
                except Exception as e:
                    st.error(f"Calibration failed: {e}")

        else:
            st.markdown("Upload CSV with `maturity, log_moneyness, implied_vol`")
            uploaded = st.file_uploader("Upload SVI Data", type=["csv"])
            if uploaded:
                try:
                    df = pd.read_csv(uploaded)
                    groups = df.groupby("maturity")
                    maturities, k_mat, iv_mat = [], [], []
                    for T, g in groups:
                        maturities.append(T)
                        k_mat.append(g["log_moneyness"].values)
                        iv_mat.append(g["implied_vol"].values)

                    surface, _ = SVI_Calibrator.calibrate_svi_surface(k_mat, iv_mat, maturities)
                    fig = PlotUtils.plot_svi_vol_surface(k_mat, maturities, surface)
                    st.plotly_chart(fig, use_container_width=True)
                except Exception as e:
                    st.error(f"Full surface calibration failed: {e}")


def render_sabr_tab(tab):
    with tab:
        st.subheader("SABR Calibration")
        mode = st.radio("Mode", ["Single Maturity", "Full Surface"])

        if mode == "Single Maturity":
            K_input = st.text_area("Strikes", "90, 100, 110")
            iv_input = st.text_area("Market IVs", "0.22, 0.21, 0.23")
            F = st.number_input("Forward Price", 100.0)
            T = st.number_input("Maturity (T)", 0.5)

            if st.button("Calibrate SABR"):
                try:
                    K = np.array([float(x) for x in K_input.split(",")])
                    iv = np.array([float(x) for x in iv_input.split(",")])
                    if len(K) != len(iv):
                        st.warning("⚠️ Length mismatch.")
                        return
                    sabr = SABRCalibrator(F, K, T, iv, beta_fixed=0.5)
                    params = sabr.calibrate()
                    fitted = sabr.model_vols()
                    for name, val in zip(["Alpha", "Beta", "Rho", "Nu"], params):
                        st.write(f"**{name}**: {val:.4f}")
                    fig = PlotUtils.plot_sabr_fit_surface(K, iv, fitted, F, T)
                    st.plotly_chart(fig, use_container_width=True)
                except Exception as e:
                    st.error(f"Calibration failed: {e}")

        else:
            uploaded = st.file_uploader("Upload SABR Data (maturity,strike,iv)", type=["csv"])
            F = st.number_input("Forward Price", 100.0)
            if uploaded:
                try:
                    df = pd.read_csv(uploaded)
                    grouped = df.groupby("maturity")
                    maturities, strike_mat, iv_mat = [], [], []
                    for T, g in grouped:
                        maturities.append(T)
                        strike_mat.append(g["strike"].values)
                        iv_mat.append(g["implied_vol"].values)

                    surface, _ = SABRCalibrator.calibrate_sabr_surface(strike_mat, iv_mat, maturities, F)
                    fig = PlotUtils.plot_sabr_vol_surface(strike_mat, maturities, surface)
                    st.plotly_chart(fig, use_container_width=True)
                except Exception as e:
                    st.error(f"Full surface calibration failed: {e}")


def render_local_vol_tab(tab):
    with tab:
        st.subheader("Local Volatility (Dupire)")

        strikes_input = st.text_area("Strikes", "80,90,100,110,120")
        maturities_input = st.text_area("Maturities (years)", "0.25,0.5,1.0")
        F = st.number_input("Forward Price", 100.0)
        uploaded = st.file_uploader("Optional IV Surface CSV", type=["csv"])

        if st.button("Generate Local Volatility Surface"):
            try:
                if uploaded:
                    iv_grid = pd.read_csv(uploaded, header=None).values
                    m, k = iv_grid.shape
                    strikes = np.linspace(80, 120, k)
                    maturities = np.linspace(0.25, 1.0, m)
                else:
                    strikes = np.array([float(x) for x in strikes_input.split(",")])
                    maturities = np.array([float(x) for x in maturities_input.split(",")])
                    iv_grid = np.array([
                        [0.2 + 0.02 * np.sin((K - 100) / 10) * np.cos(T * np.pi) for K in strikes]
                        for T in maturities
                    ])

                lv = LocalVolatilitySurface(strikes, maturities, iv_grid, F)
                local_vol = lv.generate_surface()
                fig = PlotUtils.plot_local_vol_surface(strikes, maturities, local_vol)
                st.plotly_chart(fig, use_container_width=True)
            except Exception as e:
                st.error(f"Local vol failed: {e}")


def render_heston_tab(tab):
    with tab:
        st.subheader("Heston Model (Stochastic Volatility)")

        with st.form("heston_form"):
            col1, col2 = st.columns(2)
            with col1:
                S0 = st.number_input("Spot Price", 100.0)
                T = st.number_input("Maturity (T)", 1.0)
                r = st.number_input("Risk-Free Rate", 0.01)
                opt_type = st.selectbox("Option Type", ["call", "put"])
            with col2:
                kappa = st.number_input("κ", 2.0)
                theta = st.number_input("θ", 0.04)
                sigma = st.number_input("σ", 0.5)
                rho = st.number_input("ρ", -0.7)
                v0 = st.number_input("v₀", 0.04)
            submitted = st.form_submit_button("Simulate")

        if submitted:
            try:
                fig = PlotUtils.plot_heston_price_vs_strike(S0, T, r, kappa, theta, sigma, rho, v0, opt_type)
                st.success("Simulation completed.")
                st.plotly_chart(fig, use_container_width=True)
            except Exception as e:
                st.error(f"Heston simulation error: {e}")

        st.markdown("---")
        st.subheader("Heston Calibration to Market Data")
        uploaded = st.file_uploader("Upload CSV (K,T,price)", type=["csv"])
        if uploaded:
            try:
                df = pd.read_csv(uploaded)
                if not {"K", "T", "price"}.issubset(df.columns):
                    st.error("Missing required columns.")
                    return
                data = df[["K", "T", "price"]].to_dict("records")
                with st.spinner("Calibrating..."):
                    params = calibrate_heston(data, S0, r, opt_type)
                st.success(f"Calibrated: κ={params[0]:.3f}, θ={params[1]:.3f}, σ={params[2]:.3f}, ρ={params[3]:.3f}, v₀={params[4]:.3f}")
                fig_fit = PlotUtils.plot_heston_calibration_fit(data, S0, r, params, option_type=opt_type)
                st.plotly_chart(fig_fit, use_container_width=True)
            except Exception as e:
                st.error(f"Calibration failed: {e}")
