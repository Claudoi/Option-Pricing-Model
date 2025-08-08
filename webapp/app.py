import streamlit as st
from streamlit_option_menu import option_menu

import sys, os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.ui_sections.black_scholes_ui import black_scholes_ui
from src.ui_sections.binomial_ui import binomial_ui
from src.ui_sections.monte_carlo_ui import monte_carlo_ui
from src.ui_sections.risk_ui import risk_ui
from src.ui_sections.volatility_ui import render_volatility_ui
from src.ui_sections.hedging_ui import render_hedging_ui

# --- Page config ---
st.set_page_config(
    page_title="Option Pricing Model",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Global Styles ---
st.markdown("""
    <style>
    html, body, [class*="css"]  {
        font-family: 'Inter', sans-serif;
        background-color: #121212;
        color: #F5F5F5;
    }
    .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }
    .stButton > button {
        background-color: #5E60CE;
        color: white;
        border-radius: 8px;
        padding: 0.5rem 1.5rem;
    }
    .stSelectbox > div > div {
        color: black !important;
    }
    </style>
""", unsafe_allow_html=True)

# --- App Header ---
col1, col2 = st.columns([0.9, 0.1])
with col1:
    st.markdown("""
    <h2 style='margin-bottom: 0; color: #F5F5F5;'>Option Pricing Model</h2>
    <p style='color: #AAAAAA;'>A modern quantitative finance dashboard</p>
    """, unsafe_allow_html=True)

# --- Navigation Menu ---
selected = option_menu(
    menu_title=None,
    options=["Black-Scholes", "Binomial", "Monte Carlo", "Risk Analysis", "Volatility", "Hedging"],
    icons=["calculator", "tree", "shuffle", "activity", "bar-chart", "shield"],
    orientation="horizontal",
    styles={
        "container": {"padding": "0!important", "background-color": "#1E1E1E"},
        "icon": {"color": "#AAAAAA", "font-size": "16px"},
        "nav-link": {
            "font-size": "16px",
            "text-align": "center",
            "margin": "0px",
            "color": "#F5F5F5",
            "--hover-color": "#5E60CE"
        },
        "nav-link-selected": {"background-color": "#5E60CE"},
    }
)

# --- Load Section Code ---
if selected == "Black-Scholes":
    black_scholes_ui()
elif selected == "Binomial":
    binomial_ui()
elif selected == "Monte Carlo":
    monte_carlo_ui()
elif selected == "Risk Analysis":
    risk_ui()
elif selected == "Volatility":
    render_volatility_ui()
elif selected == "Hedging":
    render_hedging_ui()