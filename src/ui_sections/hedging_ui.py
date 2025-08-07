import streamlit as st
import numpy as np
from src.utils.plot_utils import PlotUtils
from src.hedging.hedging_pnl_decomposition import HedgingPnLAttribution
from src.hedging.hedging_simulator import DeltaHedgingSimulator
from src.hedging.heston_hedging_simulator import HestonDeltaHedgingSimulator



def render_hedging_ui():
    st.markdown("## 🔐 Hedging Strategies")
    tabs = st.tabs(["Delta Hedging", "Heston Delta Hedging"])

    render_delta_hedging_tab(tabs[0])
    render_heston_hedging_tab(tabs[1])



def render_delta_hedging_tab(tab):
    with tab:
        st.subheader("Delta Hedging Simulator")
        st.markdown("Simulate dynamic delta hedging under the **Black-Scholes model**.")

        with st.form("delta_hedging_form"):
            col1, col2 = st.columns(2)
            with col1:
                S0 = st.number_input("Initial Spot Price (S₀)", value=100.0, key="dh_spot")
                K = st.number_input("Strike Price (K)", value=100.0, key="dh_strike")
                T = st.number_input("Time to Maturity (T)", value=1.0, key="dh_T")
                r = st.number_input("Risk-Free Rate (r)", value=0.01, key="dh_r")
            with col2:
                sigma = st.number_input("Volatility (σ)", value=0.2, key="dh_sigma")
                option_type = st.selectbox("Option Type", ["call", "put"], key="dh_option_type")
                steps = st.slider("Hedge Steps", 10, 365, 50, key="dh_steps")
                n_paths = st.slider("Simulated Paths", 10, 1000, 100, 10, key="dh_paths")
            submitted = st.form_submit_button("Run Delta Hedging Simulation")

        if submitted:
            try:
                simulator = DeltaHedgingSimulator(S0, K, T, r, sigma, option_type, steps, n_paths)
                pnl_paths, time_grid, pnl_over_time, hedging_errors = simulator.simulate()

                st.plotly_chart(PlotUtils.plot_hedging_pnl(time_grid, np.mean(pnl_over_time, axis=0), "📈 Mean Delta Hedging P&L"), use_container_width=True)
                st.plotly_chart(PlotUtils.plot_hedging_pnl_histogram(pnl_paths, "📊 Final P&L Distribution"), use_container_width=True)
                st.plotly_chart(PlotUtils.plot_hedging_error_over_time(time_grid, np.mean(np.abs(hedging_errors), axis=0), "📉 Mean Absolute Hedging Error"), use_container_width=True)

                st.success(f"✅ Simulation complete: Mean P&L = {np.mean(pnl_paths):.4f}, Std Dev = {np.std(pnl_paths):.4f}")

                if hasattr(simulator, "last_S"):
                    st.session_state["dh_results"] = {
                        "S": simulator.last_S, "time_grid": time_grid,
                        "K": K, "r": r, "sigma": sigma, "option_type": option_type
                    }

                    st.markdown("### 🔍 Delta Hedging P&L Decomposition")
                    decomposer = HedgingPnLAttribution(simulator.last_S, time_grid, K, r, sigma, option_type)
                    pnl_dict = decomposer.decompose()
                    st.plotly_chart(PlotUtils.plot_hedging_pnl_decomposition(pnl_dict, time_grid), use_container_width=True)
                    st.plotly_chart(PlotUtils.plot_total_pnl_cumulative(pnl_dict, time_grid), use_container_width=True)

            except Exception as e:
                st.error(f"❌ Delta hedging simulation failed: {str(e)}")



def render_heston_hedging_tab(tab):
    with tab:
        st.subheader("Heston Delta Hedging Simulator")
        st.markdown("Simulate delta hedging under the **Heston stochastic volatility model**.")

        with st.form("heston_hedging_form"):
            col1, col2 = st.columns(2)
            with col1:
                S0 = st.number_input("Spot Price", 100.0, key="heston_spot")
                K = st.number_input("Strike Price", 100.0, key="heston_strike")
                T = st.number_input("Maturity", 1.0, key="heston_T")
                r = st.number_input("Risk-Free Rate", 0.01, key="heston_r")
            with col2:
                v0 = st.number_input("Initial Variance (v₀)", 0.04, key="heston_v0")
                kappa = st.number_input("Mean Reversion Speed (κ)", 2.0, key="heston_kappa")
                theta = st.number_input("Long-Term Variance (θ)", 0.04, key="heston_theta")
                sigma_v = st.number_input("Vol Volatility (σᵥ)", 0.3, key="heston_sigma_v")
                rho = st.number_input("Correlation (ρ)", -0.7, key="heston_rho")
            steps = st.slider("Hedge Steps", 10, 365, 50, key="heston_steps")
            n_paths = st.slider("Simulated Paths", 10, 1000, 100, 10, key="heston_paths")
            submitted = st.form_submit_button("Run Heston Delta Hedging Simulation")

        if submitted:
            try:
                simulator = HestonDeltaHedgingSimulator(S0, K, T, r, v0, kappa, theta, sigma_v, rho, steps, n_paths)
                pnl_paths, time_grid, pnl_over_time, hedging_errors = simulator.simulate()

                st.plotly_chart(PlotUtils.plot_hedging_pnl(time_grid, np.mean(pnl_over_time, axis=0), "📈 Heston Delta Hedging P&L"), use_container_width=True)
                st.plotly_chart(PlotUtils.plot_hedging_pnl_histogram(pnl_paths, "📊 Final P&L Distribution (Heston)"), use_container_width=True)
                st.plotly_chart(PlotUtils.plot_hedging_error_over_time(time_grid, np.mean(np.abs(hedging_errors), axis=0), "📉 Hedging Error (Heston)"), use_container_width=True)

                st.success(f"✅ Heston simulation complete: Mean P&L = {np.mean(pnl_paths):.4f}, Std Dev = {np.std(pnl_paths):.4f}")

            except Exception as e:
                st.error(f"❌ Heston simulation failed: {str(e)}")
