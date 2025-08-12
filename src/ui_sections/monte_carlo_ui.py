import streamlit as st
import numpy as np
import plotly.graph_objs as go

from src.models.pricing_montecarlo import MonteCarloOption
from src.utils.plot_utils import PlotUtils


def _enter_section(section_name: str, keys_prefix: str, extra_keys=None):
    """
    Clear section-specific state keys when switching from another section.
    Prevents cross-contamination of widget states and results.
    """
    prev = st.session_state.get("_active_section")
    if prev != section_name:
        for k in list(st.session_state.keys()):
            if k.startswith(keys_prefix) or (extra_keys and k in extra_keys):
                st.session_state.pop(k, None)
        st.session_state["_active_section"] = section_name


def monte_carlo_ui():
    # Ensure Monte Carlo section state is isolated
    _enter_section(
        "montecarlo",
        "mc_",
        extra_keys=[
            "mc_price", "mc_paths", "mc_params", "mc_greeks", "mc_delta_comparison",
            "mc_surface_delta", "mc_surface_vega", "mc_surface_theta", "mc_surface_rho"
        ],
    )

    # --- Header ---
    st.markdown("## Monte Carlo Option Pricing")
    st.markdown('<div class="small-muted">Simulation-based pricing • Exotic payoffs • MC Greeks</div>', unsafe_allow_html=True)

    # --- Input Form ---
    st.markdown('<div class="card" style="padding:1rem;">', unsafe_allow_html=True)
    with st.form("mc_form", clear_on_submit=False):
        st.markdown("#### Option Parameters")

        # Left column - core inputs
        c1, c2 = st.columns(2)
        with c1:
            S = st.number_input("Spot Price (S)", value=100.0, min_value=0.01, format="%.2f", key="mc_S")
            K = st.number_input("Strike Price (K)", value=100.0, min_value=0.01, format="%.2f", key="mc_K")
            T = st.number_input("Time to Maturity (T, years)", value=1.0, min_value=1e-4, format="%.4f", key="mc_T")
            q = st.number_input("Dividend Yield (q)", value=0.0, min_value=0.0, format="%.4f", key="mc_q")

        # Right column - rates, vol, type, discretization
        with c2:
            r = st.number_input("Risk-Free Rate (r)", value=0.05, min_value=0.0, format="%.4f", key="mc_r")
            sigma = st.number_input("Volatility (σ)", value=0.20, min_value=1e-4, format="%.4f", key="mc_sigma")
            option_type = st.selectbox("Option Type", ["call", "put"], key="mc_option_type")
            n_sim = st.number_input("Simulations", value=10_000, min_value=1_000, step=1_000, key="mc_n_sim")
            n_steps = st.slider("Time Steps", min_value=10, max_value=500, value=100, step=10, key="mc_n_steps")

        # Exotic type selector
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
            key="mc_exotic",
        )

        # Extra inputs for barrier options only
        barrier_level = None
        barrier_type = None
        if exotic == "Digital Barrier":
            c3, c4 = st.columns(2)
            with c3:
                barrier_level = st.number_input("Barrier Level", value=float(K * 1.10), min_value=0.01, format="%.4f", key="mc_barrier_level")
            with c4:
                barrier_type = st.selectbox("Barrier Type", ["up-and-in", "up-and-out", "down-and-in", "down-and-out"], key="mc_barrier_type")

        # Optional reproducibility controls
        with st.expander("Advanced (optional)"):
            seed = st.number_input("Random Seed (reproducibility)", value=0, min_value=0, step=1, key="mc_seed")

        submitted = st.form_submit_button("Calculate")
    st.markdown('</div>', unsafe_allow_html=True)

    # --- Compute only on submit ---
    if submitted:
        if any(v <= 0 for v in [S, K, T]) or sigma <= 0:
            st.error("Inputs must be strictly positive (S, K, T, σ).")
        else:
            try:
                # Set RNG seed if requested
                if seed != 0:
                    np.random.seed(int(seed))

                # Create Monte Carlo pricer
                mc = MonteCarloOption(S, K, T, r, sigma, option_type, int(n_sim), int(n_steps), q)

                # Select payoff variant
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
                    b_lvl = float(barrier_level) if barrier_level is not None else float(K * 1.10)
                    b_typ = barrier_type if barrier_type is not None else "up-and-in"
                    price = mc.price_digital_barrier(barrier=b_lvl, barrier_type=b_typ)
                elif exotic == "American (Longstaff-Schwartz)":
                    price = mc.price_american_lsm()
                else:
                    price = None

                # Store results in session
                st.session_state["mc_price"] = price
                st.session_state["mc_paths"] = mc._simulate_paths()
                st.session_state["mc_params"] = {
                    "S": S, "K": K, "T": T, "r": r, "sigma": sigma, "option_type": option_type,
                    "n_sim": int(n_sim), "n_steps": int(n_steps), "q": q,
                    "exotic": exotic, "barrier_level": barrier_level, "barrier_type": barrier_type, "seed": int(seed),
                }
                st.session_state["mc_greeks"] = {
                    "delta": mc.greek("delta"),
                    "vega": mc.greek("vega"),
                    "theta": mc.greek("theta"),
                    "rho": mc.greek("rho"),
                }

                # Delta estimation comparison
                strikes = np.linspace(K * 0.8, K * 1.2, 9)
                delta_fd, delta_pw, delta_lr = [], [], []
                for k_ in strikes:
                    try:
                        opt = MonteCarloOption(S, float(k_), T, r, sigma, option_type, int(n_sim), int(n_steps), q)
                        delta_fd.append(opt.greek("delta"))
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

                # Clear old Greek surfaces when parameters change
                for key in ["mc_surface_delta", "mc_surface_vega", "mc_surface_theta", "mc_surface_rho"]:
                    st.session_state.pop(key, None)

            except Exception as e:
                st.error(f"Error in Monte Carlo pricing: {e}")

    # --- Exit if no results in session ---
    if "mc_params" not in st.session_state:
        st.info("Set parameters and click **Calculate** to compute simulations and Greeks.")
        return

    # --- Results summary ---
    st.markdown('<div class="card" style="padding:1rem;">', unsafe_allow_html=True)
    st.markdown("#### Results")
    p = st.session_state["mc_params"]
    cA, cB, cC, cD = st.columns(4)
    cA.metric(f"{p['exotic']} Price", f"{st.session_state['mc_price']:.4f}")
    cB.metric("Simulations", f"{p['n_sim']}")
    cC.metric("Time Steps", f"{p['n_steps']}")
    cD.metric("σ (input)", f"{p['sigma']:.4f}")
    st.markdown('</div>', unsafe_allow_html=True)

    # --- Plots and metrics ---
    st.markdown('<div class="card" style="padding:1rem;">', unsafe_allow_html=True)
    st.markdown("#### Visualizations")
    t1, t2, t3 = st.tabs(["Simulated Paths", "Payoff Distribution", "MC Greeks"])

    with t1:
        try:
            st.plotly_chart(PlotUtils.plot_mc_paths(st.session_state["mc_paths"]), use_container_width=True)
        except Exception as e:
            st.error(f"Error plotting simulated paths: {e}")

    with t2:
        try:
            ST = st.session_state["mc_paths"][:, -1]
            K_ = p["K"]
            opt_type = p["option_type"]
            payoffs = np.maximum(ST - K_, 0) if opt_type == "call" else np.maximum(K_ - ST, 0)
            fig_payoff = go.Figure(data=[go.Histogram(x=payoffs, nbinsx=50)])
            fig_payoff.update_layout(title="Payoff Distribution", xaxis_title="Payoff", yaxis_title="Frequency")
            st.plotly_chart(fig_payoff, use_container_width=True)
        except Exception as e:
            st.error(f"Error plotting payoff distribution: {e}")

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

    # --- Greek surface (persistent per Greek) ---
    st.markdown('<div class="card" style="padding:1rem;">', unsafe_allow_html=True)
    st.markdown("#### Greek Surface (Beta)")

    selected_greek = st.selectbox("Select Greek", ["delta", "vega", "theta", "rho"], key="mc_greek_surface")
    surface_key = f"mc_surface_{selected_greek}"

    # Compute surface only if not in state
    if surface_key not in st.session_state:
        grid_strikes = np.linspace(p["K"] * 0.8, p["K"] * 1.2, 10)
        grid_maturities = np.linspace(p["T"] * 0.5, p["T"] * 1.5, 10)
        greek_surface = np.zeros((len(grid_maturities), len(grid_strikes)))

        with st.spinner(f"Generating {selected_greek} surface..."):
            for i, t in enumerate(grid_maturities):
                for j, k_ in enumerate(grid_strikes):
                    try:
                        greek_surface[i, j] = MonteCarloOption(
                            p["S"], float(k_), float(t), p["r"], p["sigma"], p["option_type"],
                            p["n_sim"], p["n_steps"], p["q"]
                        ).greek(selected_greek)
                    except Exception:
                        greek_surface[i, j] = np.nan

        st.session_state[surface_key] = (grid_strikes, grid_maturities, greek_surface)

    # Always plot from session to avoid disappearing graphs
    K_grid, T_grid, surf_data = st.session_state[surface_key]
    fig_surface = PlotUtils.plot_mc_greek_surface(K_grid, T_grid, surf_data, selected_greek.capitalize())
    st.plotly_chart(fig_surface, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)
