# src/ui_sections/volatility_ui.py
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
    """Top-level UI for the Volatility section (tabs + routers)."""
    st.markdown("## Volatility Modeling & Calibration")

    tabs = st.tabs(["Vol Surface", "SVI", "SABR", "Local Vol", "Heston"])

    _render_vol_surface_tab(tabs[0])
    _render_svi_tab(tabs[1])
    _render_sabr_tab(tabs[2])
    _render_local_vol_tab(tabs[3])
    _render_heston_tab(tabs[4])


# =========================
# Vol Surface (Market Data)
# =========================
def _render_vol_surface_tab(tab):
    with tab:
        st.markdown('<div class="card" style="padding:1rem;">', unsafe_allow_html=True)
        st.subheader("Volatility Surface from Market Data")

        ticker = st.text_input("Ticker Symbol", "AAPL", key="vs_ticker_input")
        st.caption("Enter a valid ticker and click the button to fetch and plot the surface.")

        if st.button("Load Volatility Surface", key="vs_load_btn"):
            try:
                vs = VolatilitySurface(ticker)
                vs.fetch_data()  # fetch before plotting
                fig = PlotUtils.plot_market_vol_surface(vs)
                st.plotly_chart(fig, use_container_width=True)
            except Exception as e:
                st.error(f"Failed to load volatility surface: {e}")
        st.markdown('</div>', unsafe_allow_html=True)


# ==============
# SVI Calibration
# ==============
def _render_svi_tab(tab):
    with tab:
        st.markdown('<div class="card" style="padding:1rem;">', unsafe_allow_html=True)
        st.subheader("SVI Calibration")

        # Unique radio key + label to avoid collisions with other tabs
        mode = st.radio(
            "Calibration Mode (SVI)",
            ["Single Maturity (Smile)", "Full Surface"],
            key="svi_mode_radio",
            horizontal=True,
        )

        # ---- Single maturity (smile) ----
        if mode == "Single Maturity (Smile)":
            k_input = st.text_area(
                "Log-Moneyness (comma separated)",
                "0, -0.1, 0.1, 0.2",
                key="svi_k_input",
                help="Example: 0, -0.1, 0.1, 0.2",
            )
            iv_input = st.text_area(
                "Market IVs (comma separated)",
                "0.25, 0.24, 0.26, 0.27",
                key="svi_iv_input",
                help="Same count as log-moneyness.",
            )
            T = st.number_input(
                "Maturity (T, in years)",
                value=0.5,
                format="%.2f",
                key="svi_T_input",
            )

            if st.button("Calibrate SVI (Smile)", key="svi_smile_btn"):
                try:
                    k_vals = np.array([float(x.strip()) for x in k_input.split(",")])
                    iv_vals = np.array([float(x.strip()) for x in iv_input.split(",")])

                    if len(k_vals) != len(iv_vals):
                        st.warning("⚠️ Number of log-moneyness values and IVs must match.")
                    else:
                        svi = SVI_Calibrator(k_vals, iv_vals, T)
                        params, fitted_vols = svi.calibrate()

                        st.success("✅ Calibrated SVI Parameters:")
                        st.markdown(
                            f"- **a**: `{params[0]:.4f}`\n"
                            f"- **b**: `{params[1]:.4f}`\n"
                            f"- **rho**: `{params[2]:.4f}`\n"
                            f"- **m**: `{params[3]:.4f}`\n"
                            f"- **sigma**: `{params[4]:.4f}`"
                        )

                        fig = PlotUtils.plot_svi_fit(k_vals, iv_vals, fitted_vols, T)
                        st.plotly_chart(fig, use_container_width=True)

                except Exception as e:
                    st.error(f"❌ SVI calibration failed: {e}")

        # ---- Full surface ----
        else:
            st.markdown("Upload a CSV with columns: `maturity, log_moneyness, implied_vol`.")
            uploaded_file = st.file_uploader(
                "Upload SVI surface data (.csv)",
                type=["csv"],
                key="svi_surface_file",
            )

            if uploaded_file is not None:
                try:
                    df = pd.read_csv(uploaded_file)
                    grouped = df.groupby("maturity")

                    maturities, k_matrix, iv_matrix = [], [], []
                    for T, group in grouped:
                        maturities.append(float(T))
                        k_matrix.append(group["log_moneyness"].values)
                        iv_matrix.append(group["implied_vol"].values)

                    vol_surface, svi_params = SVI_Calibrator.calibrate_svi_surface(
                        k_matrix, iv_matrix, maturities
                    )
                    fig = PlotUtils.plot_svi_vol_surface(k_matrix, maturities, vol_surface)
                    st.plotly_chart(fig, use_container_width=True)

                except Exception as e:
                    st.error(f"❌ Failed to calibrate full SVI surface: {e}")
        st.markdown('</div>', unsafe_allow_html=True)


# ===============
# SABR Calibration
# ===============
def _render_sabr_tab(tab):
    with tab:
        st.markdown('<div class="card" style="padding:1rem;">', unsafe_allow_html=True)
        st.subheader("SABR Calibration")

        mode = st.radio(
            label="Calibration Mode (SABR)",
            options=["Single Maturity (Smile)", "Full Surface"],
            key="vol_sabr_mode_radio",  
            horizontal=True,
        )


        # ---- Single maturity (smile) ----
        if mode == "Single Maturity (Smile)":
            K_input = st.text_area(
                "Strikes (comma separated)",
                "90, 100, 110, 120",
                key="sabr_K_input",
            )
            iv_input = st.text_area(
                "Market IVs (comma separated)",
                "0.22, 0.21, 0.23, 0.24",
                key="sabr_iv_input",
            )
            F = st.number_input(
                "Forward Price (F)", value=100.0, key="sabr_forward_input"
            )
            T = st.number_input(
                "Maturity (T, in years)",
                value=0.5,
                format="%.2f",
                key="sabr_T_input",
            )

            if st.button("Calibrate SABR (Smile)", key="sabr_smile_btn"):
                try:
                    strikes = np.array([float(x.strip()) for x in K_input.split(",")])
                    ivs = np.array([float(x.strip()) for x in iv_input.split(",")])

                    if len(strikes) != len(ivs):
                        st.warning("⚠️ Number of strikes and IVs must match.")
                    else:
                        sabr = SABRCalibrator(F, strikes, T, ivs, beta_fixed=0.5)
                        params = sabr.calibrate()
                        fitted_vols = sabr.model_vols()

                        st.success("✅ Calibrated SABR Parameters:")
                        st.markdown(
                            f"- **Alpha**: `{params[0]:.4f}`\n"
                            f"- **Beta**: `{params[1]:.4f}`\n"
                            f"- **Rho**: `{params[2]:.4f}`\n"
                            f"- **Nu**: `{params[3]:.4f}`"
                        )

                        fig = PlotUtils.plot_sabr_fit_surface(
                            strikes, ivs, fitted_vols, F, T
                        )
                        st.plotly_chart(fig, use_container_width=True)

                except Exception as e:
                    st.error(f"❌ SABR calibration failed: {e}")

        # ---- Full surface ----
        else:
            st.markdown(
                "Upload a CSV with columns: `maturity, strike, implied_vol` to calibrate the full surface."
            )
            uploaded_file = st.file_uploader(
                "Upload SABR surface data (.csv)",
                type=["csv"],
                key="sabr_surface_file",
            )
            forward_price = st.number_input(
                "Forward Price (F)", value=100.0, key="sabr_surface_forward_input"
            )

            if uploaded_file is not None:
                try:
                    df = pd.read_csv(uploaded_file)
                    required_columns = {"maturity", "strike", "implied_vol"}
                    if not required_columns.issubset(df.columns):
                        raise ValueError(
                            "CSV must contain the columns: maturity, strike, implied_vol"
                        )

                    grouped = df.groupby("maturity")
                    maturities, strike_matrix, iv_matrix = [], [], []

                    for T, group in grouped:
                        maturities.append(float(T))
                        strike_matrix.append(group["strike"].values)
                        iv_matrix.append(group["implied_vol"].values)

                    vol_surface, sabr_params = SABRCalibrator.calibrate_sabr_surface(
                        strike_matrix, iv_matrix, maturities, forward_price
                    )
                    fig = PlotUtils.plot_sabr_vol_surface(
                        strike_matrix, maturities, vol_surface
                    )
                    st.plotly_chart(fig, use_container_width=True)

                except Exception as e:
                    st.error(f"❌ Failed to calibrate full SABR surface: {e}")
        st.markdown('</div>', unsafe_allow_html=True)


# =================
# Local Volatility
# =================
def _render_local_vol_tab(tab):
    with tab:
        st.markdown('<div class="card" style="padding:1rem;">', unsafe_allow_html=True)
        st.subheader("📉 Local Volatility Surface (Dupire)")

        # Inputs
        strikes_input = st.text_area(
            "Strikes (comma separated)",
            "80,90,100,110,120",
            key="lv_strikes_input",
        )
        maturities_input = st.text_area(
            "Maturities (comma separated, years)",
            "0.25,0.5,1.0",
            key="lv_maturities_input",
        )
        F = st.number_input("Forward Price (F)", value=100.0, key="lv_forward_input")
        iv_grid_upload = st.file_uploader(
            "Optional: Upload IV Surface CSV (maturities x strikes)",
            type=["csv"],
            key="lv_iv_file",
        )

        if st.button("Show Local Vol Surface", key="lv_show_btn"):
            try:
                # Case 1: Use uploaded CSV
                if iv_grid_upload is not None:
                    iv_grid = pd.read_csv(iv_grid_upload, header=None).values
                    m, k = iv_grid.shape
                    strikes_arr = np.linspace(80, 120, k)
                    maturities_arr = np.linspace(0.25, 1.0, m)
                    st.success(f"✅ CSV uploaded with shape ({m}, {k}).")

                # Case 2: Synthetic IV grid if nothing uploaded
                else:
                    strikes_arr = np.array(
                        [float(x.strip()) for x in strikes_input.split(",")]
                    )
                    maturities_arr = np.array(
                        [float(x.strip()) for x in maturities_input.split(",")]
                    )
                    iv_grid = np.array(
                        [
                            [
                                0.20
                                + 0.02
                                * np.sin((K - 100) / 10)
                                * np.cos(T * np.pi)
                                for K in strikes_arr
                            ]
                            for T in maturities_arr
                        ]
                    )
                    st.info("ℹ️ Generated synthetic IV surface.")

                # Build and plot local vol surface
                lv_surface = LocalVolatilitySurface(
                    strikes_arr, maturities_arr, iv_grid, F=F
                )
                local_vol_grid = lv_surface.generate_surface()
                fig = PlotUtils.plot_local_vol_surface(
                    strikes_arr, maturities_arr, local_vol_grid
                )
                if fig:
                    st.plotly_chart(fig, use_container_width=True)

            except Exception as e:
                st.error(f"❌ Failed to generate local volatility surface: {e}")
        st.markdown('</div>', unsafe_allow_html=True)


# =============
# Heston Model
# =============
def _render_heston_tab(tab):
    with tab:
        st.markdown('<div class="card" style="padding:1rem;">', unsafe_allow_html=True)
        st.subheader("Heston Model: Price vs Strike")
        st.caption("Simulate European option prices under the Heston stochastic volatility model.")

        # Parameter form
        with st.form("heston_form_unique"):
            col1, col2 = st.columns(2)
            with col1:
                S0 = st.number_input("Spot Price (S₀)", value=100.0, key="heston_S0")
                T = st.number_input("Maturity (T, in years)", value=1.0, format="%.2f", key="heston_T")
                r = st.number_input("Risk-Free Rate (r)", value=0.01, format="%.4f", key="heston_r")
                option_type = st.selectbox("Option Type", ["call", "put"], key="heston_opt_type")
            with col2:
                kappa = st.number_input("Mean Reversion Speed (κ)", value=2.0, key="heston_kappa")
                theta = st.number_input("Long-Run Variance (θ)", value=0.04, key="heston_theta")
                sigma = st.number_input("Volatility of Volatility (σ)", value=0.5, key="heston_sigma")
                rho = st.number_input("Correlation (ρ)", value=-0.7, key="heston_rho")
                v0 = st.number_input("Initial Variance (v₀)", value=0.04, key="heston_v0")

            submitted = st.form_submit_button("Simulate Heston")

        if submitted:
            try:
                fig = PlotUtils.plot_heston_price_vs_strike(
                    S0=S0, T=T, r=r,
                    kappa=kappa, theta=theta, sigma=sigma,
                    rho=rho, v0=v0, option_type=option_type
                )
                st.success("✅ Simulation completed successfully.")
                st.plotly_chart(fig, use_container_width=True)
            except Exception as e:
                st.error(f"❌ Simulation failed: {e}")

        st.markdown('</div>', unsafe_allow_html=True)

        # --- Calibration block ---
        st.markdown('<div class="card" style="padding:1rem;">', unsafe_allow_html=True)
        st.subheader("Heston Calibration: Fit to Market Data")

        uploaded = st.file_uploader(
            "Upload Market Data CSV (columns: K,T,price)",
            type=["csv"],
            key="heston_csv_file",
        )

        if uploaded is not None:
            try:
                df = pd.read_csv(uploaded)
                required_cols = {"K", "T", "price"}
                if not required_cols.issubset(df.columns):
                    st.error("❌ CSV must contain columns: 'K', 'T', 'price'")
                else:
                    market_data = df[["K", "T", "price"]].to_dict("records")
                    with st.spinner("⏳ Calibrating Heston model..."):
                        params = calibrate_heston(market_data, S0=S0, r=r, option_type=option_type)

                    st.success(
                        f"✅ Calibration completed: κ={params[0]:.3f}, θ={params[1]:.3f}, "
                        f"σ={params[2]:.3f}, ρ={params[3]:.3f}, v₀={params[4]:.3f}"
                    )

                    fig_fit = PlotUtils.plot_heston_calibration_fit(
                        market_data, S0, r, params, option_type=option_type
                    )
                    st.plotly_chart(fig_fit, use_container_width=True)

            except Exception as e:
                st.error(f"❌ Failed to process or calibrate: {e}")
        st.markdown('</div>', unsafe_allow_html=True)
