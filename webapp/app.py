import streamlit as st
from streamlit_option_menu import option_menu
import plotly.graph_objects as go

import sys, os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


from src.models.pricing_black_scholes import BlackScholesOption
from src.models.greeks import BlackScholesGreeks
from src.utils.plot_utils import plot_price_vs_spot, plot_greeks_vs_spot

# --- Configure page ---
st.set_page_config(
    page_title="Option Pricing Model",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# --- Dark/Light mode toggle ---
if "theme_mode" not in st.session_state:
    st.session_state.theme_mode = "dark"

toggle = st.toggle("", value=(st.session_state.theme_mode == "dark"),
                   help="Toggle light/dark mode")
if toggle:
    st.session_state.theme_mode = "dark"
else:
    st.session_state.theme_mode = "light"

# Apply the theme
template = "plotly_dark" if st.session_state.theme_mode == "dark" else "plotly_white"

# --- Header with toggle ---
col1, col2 = st.columns([0.9, 0.1])
with col1:
    st.markdown("## 💻 Option Pricing Interface")
with col2:
    emoji = "🌙" if st.session_state.theme_mode == "dark" else "☀️"
    st.markdown(f"### {emoji}")

# --- Option selection menu ---
selected = option_menu(
    menu_title=None,
    options=["Black-Scholes", "Binomial", "Monte Carlo", "Risk Analysis", "Volatility"],
    icons=["calculator", "tree", "shuffle", "activity", "bar-chart"],
    orientation="horizontal"
)

# --- Black-Scholes Section ---
if selected == "Black-Scholes":
    st.markdown("### Black-Scholes Option Pricing")

    with st.form("bs_inputs_form"):
        col1, col2 = st.columns(2)
        with col1:
            S = st.number_input("📈 Spot Price (S)", value=100.0, format="%.2f")
            K = st.number_input("🎯 Strike Price (K)", value=100.0, format="%.2f")
            T = st.number_input("⏳ Time to Maturity (T in years)", value=1.0, format="%.2f")
            q = st.number_input("💸 Dividend Yield (q)", value=0.0, format="%.4f")
        with col2:
            r = st.number_input("🏦 Risk-Free Rate (r)", value=0.05, format="%.4f")
            sigma = st.number_input("📊 Volatility (σ)", value=0.2, format="%.4f")
            option_type = st.selectbox("📝 Option Type", ["call", "put"])

        submitted = st.form_submit_button("💡 Calculate Option")

    if submitted:
        option = BlackScholesOption(S, K, T, r, sigma, option_type, q)
        price = option.price()
        greeks = option.greeks()

        st.success(f"💰 **Option Price**: {price:.4f}")

        st.markdown("#### 🧮 Greeks")
        g1, g2, g3 = st.columns(3)
        g1.metric("Delta", f"{greeks['delta']:.4f}")
        g2.metric("Gamma", f"{greeks['gamma']:.4f}")
        g3.metric("Vega", f"{greeks['vega']:.4f}")
        g1.metric("Theta", f"{greeks['theta']:.4f}")
        g2.metric("Rho", f"{greeks['rho']:.4f}")

        st.markdown("#### 📉 Price vs Spot & Greeks vs Spot")

        fig1 = plot_price_vs_spot(K, T, r, sigma, option_type, q, BlackScholesOption)
        fig2 = plot_greeks_vs_spot(K, T, r, sigma, option_type, q, BlackScholesOption)

        st.plotly_chart(fig1, use_container_width=True)
        st.plotly_chart(fig2, use_container_width=True)
