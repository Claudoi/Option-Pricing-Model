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
    page_icon="assets/icon.png",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Theme-aware Global Styles (no hardcoded colors) ---
st.markdown("""
<style>
/* Select container */
.stSelectbox > div { 
  background: var(--background-color); 
  border: 1px solid var(--secondary-background-color); 
  border-radius: 10px;
}
/* Select text + icons */
.stSelectbox [data-baseweb="select"] * { 
  color: var(--text-color) !important;
}
.stSelectbox [data-baseweb="select"] svg { 
  fill: var(--text-color) !important;
}
/* Dropdown popover */
[data-baseweb="popover"]{
  background: var(--background-color); 
  border: 1px solid var(--secondary-background-color); 
  border-radius: 10px;
}
[data-baseweb="popover"] * { 
  color: var(--text-color) !important;
}

/* Header text (inherit theme) */
.app-header-title { color: var(--text-color); margin-bottom: 0; }
.app-header-sub   { color: rgba( var(--color-text-rgb, 255,255,255), 0.65 ); margin-top: 2px; }

/* Option menu (streamlit_option_menu) — theme-aware */
.om-container { 
  padding: 0 !important; 
  background-color: var(--background-color);
  border-bottom: 1px solid var(--secondary-background-color);
  border-radius: 8px;
}
.om-icon    { color: var(--text-color); }
.om-link    { color: var(--text-color); }
.om-link:hover { 
  background-color: var(--secondary-background-color);
}
.om-selected { 
  background-color: var(--primary-color); 
  color: var(--background-color) !important;
}
</style>
""", unsafe_allow_html=True)

# --- App Header (no inline hardcoded colors) ---
col1, col2 = st.columns([0.9, 0.1])
with col1:
    st.markdown("""
    <h2 class='app-header-title'>Option Pricing Model</h2>
    <p class='app-header-sub'>A modern quantitative finance dashboard</p>
    """, unsafe_allow_html=True)

# --- Navigation Menu (use CSS classes to avoid fixed colors) ---
selected = option_menu(
    menu_title=None,
    options=["Black-Scholes", "Binomial", "Monte Carlo", "Risk Analysis", "Volatility", "Hedging"],
    icons=["calculator", "tree", "shuffle", "activity", "bar-chart", "shield"],
    orientation="horizontal",
    styles={
        # We keep these minimal and offload actual colors to the CSS above
        "container": {"padding": "0!important"},
        "icon": {"font-size": "16px"},                # color via .om-icon
        "nav-link": {"font-size": "16px", "text-align": "center", "margin": "0px"},  # color via .om-link
        "nav-link-selected": {},                      # bg via .om-selected
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
