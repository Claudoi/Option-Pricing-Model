import streamlit as st
import numpy as np
from src.pricing_black_scholes import BlackScholesOption
from src.pricing_montecarlo import MonteCarloOption
from src.pricing_binomial import BinomialOption
from src.greeks import BlackScholesGreeks


class OptionPricingApp:
    def __init__(self):
        self.setup_ui()

    def setup_ui(self):
        st.title("\U0001F4C8 Option Pricing Interface")
        st.markdown("""
        Value European and exotic options using different pricing models:
        - Black-Scholes (analytical)
        - Monte Carlo (simulation)
        - Binomial (tree-based, European and American)
        """)
        self.select_model()
        self.collect_inputs()
        self.calculate()

    def select_model(self):
        self.model = st.selectbox("Select pricing model", [
            "Black-Scholes",
            "Monte Carlo",
            "Binomial (European)",
            "Binomial (American)"
        ])

    def collect_inputs(self):
        with st.form("input_form"):
            col1, col2 = st.columns(2)
            with col1:
                self.S = st.number_input("Spot price (S)", value=100.0)
                self.K = st.number_input("Strike price (K)", value=100.0)
                self.T = st.number_input("Time to maturity (T in years)", value=1.0)
                self.option_type = st.selectbox("Option type", ["call", "put"])
            with col2:
                self.r = st.number_input("Risk-free rate (r)", value=0.05)
                self.sigma = st.number_input("Volatility (σ)", value=0.2)
                self.q = st.number_input("Dividend yield (q)", value=0.0)

            if self.model == "Monte Carlo":
                self.n_sim = st.slider("Number of simulations", 1000, 100000, 10000, step=1000)
                self.n_steps = st.slider("Number of steps", 10, 500, 100, step=10)
            elif "Binomial" in self.model:
                self.N = st.slider("Number of binomial steps", 10, 500, 100, step=10)

            self.submitted = st.form_submit_button("\U0001F4CA Calculate")

    def calculate(self):
        if not self.submitted:
            return

        try:
            if self.model == "Black-Scholes":
                opt = BlackScholesOption(self.S, self.K, self.T, self.r, self.sigma, self.option_type, self.q)
                price = opt.price()
                greeks = opt.greeks()

                st.success(f"Black-Scholes Price: {price:.4f}")
                st.markdown("**Greeks:**")
                st.json(greeks)

                # Optional: Add explicit greeks from greeks.py
                st.markdown("**Explicit Greeks (from BlackScholesGreeks):**")
                greek_model = BlackScholesGreeks(self.S, self.K, self.T, self.r, self.sigma, self.option_type)
                st.json({
                    "delta": greek_model.delta(),
                    "gamma": greek_model.gamma(),
                    "vega": greek_model.vega(),
                    "theta": greek_model.theta(),
                    "rho": greek_model.rho()
                })

            elif self.model == "Monte Carlo":
                mc = MonteCarloOption(self.S, self.K, self.T, self.r, self.sigma, self.option_type, self.n_sim, self.n_steps, self.q)
                price = mc.price_vanilla()
                st.success(f"Monte Carlo Price: {price:.4f}")

            elif self.model == "Binomial (European)":
                bopt = BinomialOption(self.S, self.K, self.T, self.r, self.sigma, self.N, self.option_type, self.q)
                price = bopt.price_european()
                st.success(f"Binomial European Price: {price:.4f}")

            elif self.model == "Binomial (American)":
                bopt = BinomialOption(self.S, self.K, self.T, self.r, self.sigma, self.N, self.option_type, self.q)
                price = bopt.price_american()
                st.success(f"Binomial American Price: {price:.4f}")

        except Exception as e:
            st.error(f"Error during calculation: {str(e)}")


if __name__ == "__main__":
    app = OptionPricingApp()
    app.run()

