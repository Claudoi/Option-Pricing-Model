import streamlit as st
import numpy as np
import plotly.graph_objs as go

from src.models.pricing_montecarlo import MonteCarloOption
from src.utils.plot_utils import PlotUtils


def monte_carlo_ui():
    
    # Header
    st.markdown("## Monte Carlo Option Pricing")
    st.markdown('<div class="small-muted">Simulation-based pricing • Exotic payoffs • MC Greeks</div>', unsafe_allow_html=True)

    # Input Form
    st.markdown('<div class="card" style="padding:1rem;">', unsafe_allow_html=True)
    with st.form("mc_form"):
        st.markdown("#### Option Parameters")

        # Core params
        c1, c2 = st.columns(2)
        with c1:
            S = st.number_input("Spot Price (S)", value=100.0, min_value=0.01, format="%.2f")
            K = st.number_input("Strike Price (K)", value=100.0, min_value=0.01, format="%.2f")
            T = st.number_input("Time to Maturity (T, years)", value=1.0, min_value=1e-4, format="%.4f")
            q = st.number_input("Dividend Yield (q)", value=0.0, min_value=0.0, format="%.4f")

        with c2:
            r = st.number_input("Risk-Free Rate (r)", value=0.05, min_value=0.0, format="%.4f")
            sigma = st.number_input("Volatility (σ)", value=0.20, min_value=1e-4, format="%.4f")
            option_type = st.selectbox("Option Type", ["call", "put"])
            n_sim = st.number_input("Simulations", value=10_000, min_value=1_000, step=1_000)
            n_steps = st.slider("Time Steps", min_value=10, max_value=500, value=100, step=10)

        # Exotic selector + variant-specific controls
        exotic = st.selectbox(
            "Option Variant",
            [
                "Vanilla",
                "Asian (arithmetic)",
                "Asian (geometric)",
                "Lookback (fixed)",
                "Lookback (floating)",
                "Digital Barrier",
                "American (Longstaff-Schwartz)",
            ],
        )

        # Variant-specific UI (barrier level/type only when needed)
        barrier_level = None
        barrier_type = None
        if exotic == "Digital Barrier":
            c3, c4 = st.columns(2)
            with c3:
                barrier_level = st.number_input("Barrier Level", value=float(K * 1.10), min_value=0.01, format="%.4f",
                                                help="Typical examples: up/down relative to spot or strike.")
            with c4:
                barrier_type = st.selectbox(
                    "Barrier Type",
                    ["up-and-in", "up-and-out", "down-and-in", "down-and-out"],
                    help="Choose whether the option knocks in or out and the barrier direction.",
                )

        # Optional: reproducibility
        with st.expander("Advanced (optional)"):
            seed = st.number_input("Random Seed (reproducibility)", value=0, min_value=0, step=1,
                                   help="Set 0 to let the RNG choose; any other value fixes the seed.")

        submitted = st.form_submit_button("Calculate")
    st.markdown('</div>', unsafe_allow_html=True)

    # Early Exit
    if not submitted:
        return

    # Basic Validation
    if any(v <= 0 for v in [S, K, T]) or sigma <= 0:
        st.error("Inputs must be strictly positive (S, K, T, σ).")
        return
    if n_sim < 1000 or n_steps < 10:
        st.warning("For stable estimates, consider ≥ 1,000 sims and ≥ 10 time steps.")

    # Pricing & Path Simulation
    try:
        # Set seed (if provided)
        if seed != 0:
            np.random.seed(int(seed))

        mc = MonteCarloOption(S, K, T, r, sigma, option_type, int(n_sim), int(n_steps), q)

        # Price by variant
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
            # Fallback default in case user left it blank (shouldn't happen due to UI)
            b_lvl = float(barrier_level) if barrier_level is not None else float(K * 1.10)
            b_typ = barrier_type if barrier_type is not None else "up-and-in"
            price = mc.price_digital_barrier(barrier=b_lvl, barrier_type=b_typ)
        elif exotic == "American (Longstaff-Schwartz)":
            price = mc.price_american_lsm()
        else:
            price = None

        # Persist results in session (for later tabs/cards)
        st.session_state["mc_price"] = price
        # Simulate and store paths (if your class exposes a public method, prefer it over _simulate_paths)
        st.session_state["mc_paths"] = mc._simulate_paths()
        st.session_state["mc_params"] = {
            "S": S, "K": K, "T": T, "r": r, "sigma": sigma, "option_type": option_type,
            "n_sim": int(n_sim), "n_steps": int(n_steps), "q": q,
            "exotic": exotic,
            "barrier_level": barrier_level,
            "barrier_type": barrier_type,
            "seed": int(seed),
        }

        # Greeks (Monte Carlo estimates)
        st.session_state["mc_greeks"] = {
            "delta": mc.greek("delta"),
            "vega": mc.greek("vega"),
            "theta": mc.greek("theta"),
            "rho": mc.greek("rho"),
        }

        # Delta comparison across strikes (FD vs Pathwise vs Likelihood Ratio)
        strikes = np.linspace(K * 0.8, K * 1.2, 9)
        delta_fd, delta_pw, delta_lr = [], [], []
        for k_ in strikes:
            try:
                opt = MonteCarloOption(S, float(k_), T, r, sigma, option_type, int(n_sim), int(n_steps), q)
                delta_fd.append(opt.greek("delta"))
                # Only call methods that exist in your class
                delta_pw.append(opt.pathwise_delta())
                delta_lr.append(opt.likelihood_ratio_delta())
            except Exception:
                delta_fd.append(np.nan); delta_pw.append(np.nan); delta_lr.append(np.nan)

        st.session_state["mc_delta_comparison"] = {
            "strikes": strikes,
            "Finite Diff": delta_fd,
            "Pathwise": delta_pw,
            "Likelihood Ratio": delta_lr,
        }

    except Exception as e:
        st.error(f"Error in Monte Carlo pricing: {e}")
        return

    # Results Card
    st.markdown('<div class="card" style="padding:1rem;">', unsafe_allow_html=True)
    st.markdown("#### Results")
    cA, cB, cC, cD = st.columns(4)
    cA.metric(f"{st.session_state['mc_params']['exotic']} Price", f"{st.session_state['mc_price']:.4f}")
    cB.metric("Simulations", f"{st.session_state['mc_params']['n_sim']}")
    cC.metric("Time Steps", f"{st.session_state['mc_params']['n_steps']}")
    cD.metric("σ (input)", f"{st.session_state['mc_params']['sigma']:.4f}")
    st.markdown('</div>', unsafe_allow_html=True)

    # Visualizations 
    st.markdown('<div class="card" style="padding:1rem;">', unsafe_allow_html=True)
    st.markdown("#### Visualizations")
    t1, t2, t3 = st.tabs(["Simulated Paths", "Payoff Distribution", "MC Greeks"])

    # Simulated Paths 
    with t1:
        try:
            st.plotly_chart(PlotUtils.plot_mc_paths(st.session_state["mc_paths"]), use_container_width=True)
        except Exception as e:
            st.error(f"Error plotting simulated paths: {e}")

    # Payoff Histogram 
    with t2:
        try:
            ST = st.session_state["mc_paths"][:, -1]
            K_ = st.session_state["mc_params"]["K"]
            opt_type = st.session_state["mc_params"]["option_type"]
            payoffs = np.maximum(ST - K_, 0) if opt_type == "call" else np.maximum(K_ - ST, 0)

            fig_payoff = go.Figure(data=[go.Histogram(x=payoffs, nbinsx=50)])
            fig_payoff.update_layout(title="Payoff Distribution", xaxis_title="Payoff", yaxis_title="Frequency")
            st.plotly_chart(fig_payoff, use_container_width=True)
        except Exception as e:
            st.error(f"Error plotting payoff distribution: {e}")

    # MC Greeks
    with t3:
        try:
            g = st.session_state["mc_greeks"]
            g1, g2, g3 = st.columns(3)
            g1.metric("Delta", f"{g['delta']:.4f}")
            g2.metric("Vega", f"{g['vega']:.4f}")
            g3.metric("Theta", f"{g['theta']:.4f}")
            g4, _ = st.columns(2)
            g4.metric("Rho", f"{g['rho']:.4f}")
        except Exception as e:
            st.error(f"Error presenting MC Greeks: {e}")

    st.markdown('</div>', unsafe_allow_html=True)

    # Delta Methods Comparison 
    st.markdown('<div class="card" style="padding:1rem;">', unsafe_allow_html=True)
    st.markdown("#### Delta Estimation Methods Comparison")
    try:
        dc = st.session_state["mc_delta_comparison"]
        greek_dict = {
            "Finite Diff": dc["Finite Diff"],
            "Pathwise": dc["Pathwise"],
            "Likelihood Ratio": dc["Likelihood Ratio"],
        }
        st.plotly_chart(
            PlotUtils.plot_mc_greek_comparison(
                dc["strikes"], greek_dict, "Delta", "Delta Estimation via Monte Carlo"
            ),
            use_container_width=True,
        )
    except Exception as e:
        st.error(f"Error plotting delta comparison: {e}")
    st.markdown('</div>', unsafe_allow_html=True)


    # Greek Surface 
    st.markdown('<div class="card" style="padding:1rem;">', unsafe_allow_html=True)
    st.markdown("#### Greek Surface (Beta)")
    selected_greek = st.selectbox("Select Greek", ["delta", "vega", "theta", "rho"], key="mc_greek_surface")

    try:
        p = st.session_state["mc_params"]
        grid_strikes = np.linspace(p["K"] * 0.8, p["K"] * 1.2, 10)
        grid_maturities = np.linspace(p["T"] * 0.5, p["T"] * 1.5, 10)
        greek_surface = np.zeros((len(grid_maturities), len(grid_strikes)))

        with st.spinner("Generating 3D Surface..."):
            for i, t in enumerate(grid_maturities):
                for j, k_ in enumerate(grid_strikes):
                    try:
                        greek_surface[i, j] = MonteCarloOption(
                            p["S"], float(k_), float(t), p["r"], p["sigma"], p["option_type"],
                            p["n_sim"], p["n_steps"], p["q"]
                        ).greek(selected_greek)
                    except Exception:
                        greek_surface[i, j] = np.nan

        fig_surface = PlotUtils.plot_mc_greek_surface(
            grid_strikes, grid_maturities, greek_surface, selected_greek.capitalize()
        )
        st.plotly_chart(fig_surface, use_container_width=True)
    except Exception as e:
        st.error(f"Error generating Greek surface: {e}")
    st.markdown('</div>', unsafe_allow_html=True)
