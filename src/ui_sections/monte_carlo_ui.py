import streamlit as st
import numpy as np
import plotly.graph_objs as go

from src.models.pricing_montecarlo import MonteCarloOption
from src.utils.plot_utils import PlotUtils


def monte_carlo_ui():
    st.markdown("### 🎲 Monte Carlo Option Pricing")

    with st.form("mc_form"):
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
            n_sim = st.number_input("Simulations", value=10000, min_value=1000, step=1000)
            n_steps = st.slider("Time Steps", min_value=10, max_value=500, value=100, step=10)
            exotic = st.selectbox(
                "Option Variant",
                ["Vanilla", "Asian (arithmetic)", "Asian (geometric)",
                 "Lookback (fixed)", "Lookback (floating)",
                 "Digital Barrier", "American (Longstaff-Schwartz)"]
            )
        submitted = st.form_submit_button("Calculate")

    if submitted:
        try:
            mc = MonteCarloOption(S, K, T, r, sigma, option_type, n_sim, n_steps, q)

            # Price according to variant
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

            # Store data in session
            st.session_state["mc_price"] = price
            st.session_state["mc_paths"] = mc._simulate_paths()
            st.session_state["mc_params"] = {
                "S": S, "K": K, "T": T, "r": r, "sigma": sigma, "option_type": option_type,
                "n_sim": n_sim, "n_steps": n_steps, "q": q
            }

            # Store Greeks
            st.session_state["mc_greeks"] = {
                "delta": mc.greek("delta"),
                "vega": mc.greek("vega"),
                "theta": mc.greek("theta"),
                "rho": mc.greek("rho")
            }

            # Delta comparison
            strikes = np.linspace(K * 0.8, K * 1.2, 9)
            delta_fd, delta_pw, delta_lr = [], [], []
            for k_ in strikes:
                try:
                    opt = MonteCarloOption(S, k_, T, r, sigma, option_type, n_sim, n_steps, q)
                    delta_fd.append(opt.greek("delta"))
                    delta_pw.append(opt.pathwise_delta())
                    delta_lr.append(opt.likelihood_ratio_delta())
                except:
                    delta_fd.append(np.nan); delta_pw.append(np.nan); delta_lr.append(np.nan)

            st.session_state["mc_delta_comparison"] = {
                "strikes": strikes,
                "Finite Diff": delta_fd,
                "Pathwise": delta_pw,
                "Likelihood Ratio": delta_lr
            }

        except Exception as e:
            st.error(f"Error in Monte Carlo pricing: {e}")

    # --- Display results if stored ---
    if "mc_price" in st.session_state:
        st.success(f"Monte Carlo {exotic} Price: {st.session_state['mc_price']:.4f}")

        st.markdown("#### 🧪 Simulated Price Paths")
        st.plotly_chart(PlotUtils.plot_mc_paths(st.session_state["mc_paths"]), use_container_width=True)

        # Histogram
        ST = st.session_state["mc_paths"][:, -1]
        K = st.session_state["mc_params"]["K"]
        option_type = st.session_state["mc_params"]["option_type"]
        payoffs = np.maximum(ST - K, 0) if option_type == "call" else np.maximum(K - ST, 0)

        fig_payoff = go.Figure(data=[go.Histogram(x=payoffs, nbinsx=50)])
        fig_payoff.update_layout(title="Payoff Distribution", xaxis_title="Payoff", yaxis_title="Frequency")
        st.plotly_chart(fig_payoff, use_container_width=True)

        # Greeks
        st.markdown("#### 📐 Greeks (Estimated)")
        col1, col2, col3 = st.columns(3)
        g = st.session_state["mc_greeks"]
        col1.metric("Delta", f"{g['delta']:.4f}")
        col2.metric("Vega", f"{g['vega']:.4f}")
        col3.metric("Theta", f"{g['theta']:.4f}")
        col1.metric("Rho", f"{g['rho']:.4f}")

        # Delta Comparison
        st.markdown("#### 🔍 Delta Estimation Methods Comparison")
        dc = st.session_state["mc_delta_comparison"]
        greek_dict = {
            "Finite Diff": dc["Finite Diff"],
            "Pathwise": dc["Pathwise"],
            "Likelihood Ratio": dc["Likelihood Ratio"]
        }
        st.plotly_chart(
            PlotUtils.plot_mc_greek_comparison(dc["strikes"], greek_dict, "Delta", "Delta Estimation via Monte Carlo"),
            use_container_width=True
        )

    # --- Optional Greek Surface ---
    if "mc_params" in st.session_state:
        st.markdown("#### 🧮 Greek Surface (Beta)")
        selected_greek = st.selectbox("Select Greek", ["delta", "vega", "theta", "rho"], key="mc_greek_surface")

        p = st.session_state["mc_params"]
        grid_strikes = np.linspace(p["K"] * 0.8, p["K"] * 1.2, 10)
        grid_maturities = np.linspace(p["T"] * 0.5, p["T"] * 1.5, 10)
        greek_surface = np.zeros((len(grid_maturities), len(grid_strikes)))

        with st.spinner("Generating 3D Surface..."):
            for i, t in enumerate(grid_maturities):
                for j, k_ in enumerate(grid_strikes):
                    try:
                        greek_surface[i, j] = MonteCarloOption(
                            p["S"], k_, t, p["r"], p["sigma"], p["option_type"],
                            p["n_sim"], p["n_steps"], p["q"]
                        ).greek(selected_greek)
                    except:
                        greek_surface[i, j] = np.nan

        fig_surface = PlotUtils.plot_mc_greek_surface(grid_strikes, grid_maturities, greek_surface, selected_greek.capitalize())
        st.plotly_chart(fig_surface, use_container_width=True)
