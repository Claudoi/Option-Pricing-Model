import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import sys
import os
import pandas as pd

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.pricing_black_scholes import BlackScholesOption
from src.pricing_montecarlo import MonteCarloOption
from src.pricing_binomial import BinomialOption
from src.greeks import BlackScholesGreeks
from mpl_toolkits.mplot3d import Axes3D


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
            "Binomial (American)",
            "Risk Analysis"
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


            # Optional inputs for specific models
            if self.model == "Monte Carlo":
                self.exotic_type = st.selectbox("Exotic option type", [
                    "Vanilla",
                    "Asian (arithmetic)",
                    "Asian (geometric)",
                    "American (Longstaff-Schwartz)",
                    "Lookback (fixed)",
                    "Lookback (floating)",
                    "Digital Barrier (up-and-in)"
                ])
                self.n_sim = st.slider("Number of simulations", 1000, 100000, 10000, step=1000)
                self.n_steps = st.slider("Number of steps", 10, 500, 100, step=10)


            elif "Binomial" in self.model:
                self.N = st.slider("Number of binomial steps", 10, 500, 100, step=10)


            # Optional implied volatility estimation (only for Black-Scholes)
            elif self.model == "Black-Scholes":
                self.use_iv = st.checkbox("Estimate implied volatility from market price")
                if self.use_iv:
                    self.market_price = st.number_input("Market option price", min_value=0.01, value=10.0)

        
            elif self.model == "Risk Analysis":
                self.risk_method = st.selectbox("Risk method", ["Parametric", "Historical", "Monte Carlo"])
                self.confidence_level = st.slider("Confidence Level", 0.80, 0.99, 0.95, step=0.01)
                self.holding_period = st.number_input("Holding period (days)", min_value=1, value=1)
                self.tickers = st.text_input("Enter asset tickers (comma-separated)", value="AAPL,MSFT,GOOGL")
                self.start_date = st.date_input("Start date")
                self.end_date = st.date_input("End date")
                
                if self.risk_method == "Monte Carlo":
                    self.n_sim = st.slider("Number of simulations", 1000, 50000, 10000, step=1000)


            self.submitted = st.form_submit_button("\U0001F4CA Calculate")


    def plot_paths(self, paths):
        fig, ax = plt.subplots(figsize=(8, 4))
        for i in range(min(50, len(paths))):
            ax.plot(paths[i], lw=0.5)
        ax.set_title("Monte Carlo Simulated Paths")
        ax.set_xlabel("Step")
        ax.set_ylabel("Price")
        st.pyplot(fig)

    def calculate(self):
        if not self.submitted:
            return

        try:
            if self.model == "Black-Scholes":
                # If implied volatility estimation is enabled
                if getattr(self, "use_iv", False):
                    try:
                        implied_vol = BlackScholesOption.implied_volatility_newton(
                            market_price=self.market_price,
                            S=self.S,
                            K=self.K,
                            T=self.T,
                            r=self.r,
                            option_type=self.option_type,
                            q=self.q
                        )
                        st.success(f"Implied Volatility: {implied_vol:.4%}")
                        self.sigma = implied_vol
                    except Exception as e:
                        st.warning(f"Could not compute implied volatility: {str(e)}")

                # Create Black-Scholes option object
                opt = BlackScholesOption(self.S, self.K, self.T, self.r, self.sigma, self.option_type, self.q)
                price = opt.price()
                greeks = opt.greeks()

                # Display results
                st.success(f"Black-Scholes Price: {price:.4f}")
                st.markdown("### Greeks (from BlackScholesOption)")
                col1, col2, col3 = st.columns(3)
                col1.metric("Delta", round(greeks["delta"], 4))
                col2.metric("Gamma", round(greeks["gamma"], 4))
                col3.metric("Vega", round(greeks["vega"], 4))
                col1.metric("Theta", round(greeks["theta"], 4))
                col2.metric("Rho", round(greeks["rho"], 4))

                st.markdown("### Explicit Greeks (BlackScholesGreeks)")
                greek_model = BlackScholesGreeks(self.S, self.K, self.T, self.r, self.sigma, self.option_type)
                st.json({
                    "delta": greek_model.delta(),
                    "gamma": greek_model.gamma(),
                    "vega": greek_model.vega(),
                    "theta": greek_model.theta(),
                    "rho": greek_model.rho()
                })

                # Save inputs for future plotting
                st.session_state.submitted = True
                st.session_state._last_S = self.S
                st.session_state._last_K = self.K
                st.session_state._last_T = self.T
                st.session_state._last_r = self.r
                st.session_state._last_sigma = self.sigma
                st.session_state._last_option_type = self.option_type
                st.session_state._last_q = self.q




            elif self.model == "Monte Carlo":
                mc = MonteCarloOption(self.S, self.K, self.T, self.r, self.sigma, self.option_type,
                                    self.n_sim, self.n_steps, self.q)

                # Map exotic option types to pricing methods
                exotic_pricers = {
                    "Vanilla": lambda: mc.price_vanilla(),
                    "Asian (arithmetic)": lambda: mc.price_asian(),
                    "Asian (geometric)": lambda: mc.price_asian_geometric(),
                    "American (Longstaff-Schwartz)": lambda: mc.price_american_lsm(),
                    "Lookback (fixed)": lambda: mc.price_lookback(strike_type="fixed"),
                    "Lookback (floating)": lambda: mc.price_lookback(strike_type="floating"),
                    "Digital Barrier (up-and-in)": lambda: mc.price_digital_barrier(barrier=self.K * 1.1, barrier_type="up-and-in"),
                    "Digital Barrier (up-and-out)": lambda: mc.price_digital_barrier(barrier=self.K * 1.1, barrier_type="up-and-out"),
                    "Digital Barrier (down-and-in)": lambda: mc.price_digital_barrier(barrier=self.K * 0.9, barrier_type="down-and-in"),
                    "Digital Barrier (down-and-out)": lambda: mc.price_digital_barrier(barrier=self.K * 0.9, barrier_type="down-and-out")
                }

                if self.exotic_type in exotic_pricers:
                    price = exotic_pricers[self.exotic_type]()
                    paths = mc._simulate_paths()
                    self.plot_paths(paths)
                    st.success(f"Monte Carlo {self.exotic_type} Price: {price:.4f}")
                else:
                    st.warning(f"Exotic option type '{self.exotic_type}' is not implemented.")



            elif self.model == "Binomial (European)":
                bopt = BinomialOption(self.S, self.K, self.T, self.r, self.sigma, self.N, self.option_type, self.q)
                price = bopt.price_european()
                st.success(f"Binomial European Price: {price:.4f}")



            elif self.model == "Binomial (American)":
                bopt = BinomialOption(self.S, self.K, self.T, self.r, self.sigma, self.N, self.option_type, self.q)
                price = bopt.price_american()
                st.success(f"Binomial American Price: {price:.4f}")



            elif self.model == "Risk Analysis":
                if not self.tickers:
                    st.warning("Please enter at least one ticker.")
                    return

                try:
                    tickers_list = [t.strip().upper() for t in self.tickers.split(",")]
                    from src.utils import fetch_returns_from_yahoo

                    df_returns = fetch_returns_from_yahoo(tickers_list, str(self.start_date), str(self.end_date))
                    returns = df_returns.values
                    assets = df_returns.columns
                    n_assets = len(assets)

                    # Equal weights if no other information is provided
                    weights = np.ones(n_assets) / n_assets

                    if self.risk_method == "Parametric":
                        from src.risk_analysis import PortfolioVaR
                        model = PortfolioVaR(pd.DataFrame(returns, columns=assets), weights,
                                            confidence_level=self.confidence_level,
                                            holding_period=self.holding_period)
                        var = model.calculate_var()
                        st.success(f"Parametric VaR: {var:.4f}")


                    elif self.risk_method == "Historical":
                        from src.risk_analysis import HistoricalVaR
                        portfolio_returns = returns @ weights
                        model = HistoricalVaR(portfolio_returns, confidence_level=self.confidence_level)
                        var = model.calculate_var()
                        es = model.calculate_es()
                        st.success(f"Historical VaR: {var:.4f}")
                        st.info(f"Expected Shortfall (ES): {es:.4f}")


                        # Histogram of losses
                        fig, ax = plt.subplots()
                        ax.hist(portfolio_returns, bins=50, color="skyblue", edgecolor="black", alpha=0.7)
                        ax.axvline(-var, color="red", linestyle="--", label=f"VaR ({self.confidence_level:.0%})")
                        ax.axvline(-es, color="orange", linestyle="--", label="ES")
                        ax.set_title("Historical Portfolio Return Distribution")
                        ax.set_xlabel("Return")
                        ax.set_ylabel("Frequency")
                        ax.legend()
                        st.pyplot(fig)


                    elif self.risk_method == "Monte Carlo":
                        from src.risk_analysis import MonteCarloVaR
                        mu = returns.mean(axis=0)
                        sigma = returns.std(axis=0)
                        S0 = df_returns.iloc[-1].values

                        model = MonteCarloVaR(
                            S0, mu, sigma, weights,
                            T=self.holding_period / 252,
                            confidence_level=self.confidence_level,
                            n_sim=self.n_sim
                        )
                        var, es, simulated_returns = model.calculate_var_es()
                        st.success(f"Monte Carlo VaR: {var:.4f}")
                        st.info(f"Expected Shortfall (ES): {es:.4f}")

                        # Visualization of simulated losses
                        fig, ax = plt.subplots()
                        ax.hist(simulated_returns, bins=50, color="skyblue", edgecolor="black", alpha=0.7)
                        ax.axvline(-var, color="red", linestyle="--", label=f"VaR ({self.confidence_level:.0%})")
                        ax.axvline(-es, color="orange", linestyle="--", label="ES")
                        ax.set_title("Monte Carlo Simulated Portfolio Returns")
                        ax.set_xlabel("Return")
                        ax.set_ylabel("Frequency")
                        ax.legend()
                        st.pyplot(fig)

                except Exception as e:
                   st.error("⚠️ Error during risk analysis.")
                   st.code(str(e))
                   st.info("🔎 Please check:\n- That all tickers are valid.\n- That the date range includes trading days.\n- That the API returned price data.")



        except Exception as e:
            st.error(f"Error during calculation: {str(e)}")


def plot_price_vs_spot(K, T, r, sigma, option_type, q):
    S_range = np.linspace(50, 150, 100)
    prices = [BlackScholesOption(S, K, T, r, sigma, option_type, q).price() for S in S_range]

    fig, ax = plt.subplots()
    ax.plot(S_range, prices)
    ax.set_title("Option Price vs Spot Price")
    ax.set_xlabel("Spot Price (S)")
    ax.set_ylabel("Option Price")
    st.pyplot(fig)


def plot_greeks_vs_spot(K, T, r, sigma, option_type, q):
    S_range = np.linspace(50, 150, 100)
    deltas, gammas = [], []

    for S in S_range:
        opt = BlackScholesOption(S, K, T, r, sigma, option_type, q)
        greeks = opt.greeks()
        deltas.append(greeks["delta"])
        gammas.append(greeks["gamma"])

    fig, ax = plt.subplots()
    ax.plot(S_range, deltas, label="Delta")
    ax.plot(S_range, gammas, label="Gamma")
    ax.set_title("Greeks vs Spot Price")
    ax.set_xlabel("Spot Price (S)")
    ax.set_ylabel("Value")
    ax.legend()
    st.pyplot(fig)


def plot_implied_vol_surface(S, r, option_type, q):
    K_vals = np.linspace(80, 120, 20)
    T_vals = np.linspace(0.1, 2.0, 20)
    K_grid, T_grid = np.meshgrid(K_vals, T_vals)
    IV_grid = np.zeros_like(K_grid)

    for i in range(K_grid.shape[0]):
        for j in range(K_grid.shape[1]):
            K = K_grid[i, j]
            T = T_grid[i, j]
            try:
                market_price = BlackScholesOption(S, K, T, r, 0.2, option_type, q).price()
                iv = BlackScholesOption.implied_volatility_newton(market_price, S, K, T, r, option_type, q)
                IV_grid[i, j] = iv
            except Exception:
                IV_grid[i, j] = np.nan

    fig = plt.figure(figsize=(10, 6))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(K_grid, T_grid, IV_grid, cmap="viridis")
    ax.set_title("Implied Volatility Surface")
    ax.set_xlabel("Strike (K)")
    ax.set_ylabel("Maturity (T)")
    ax.set_zlabel("IV")
    st.pyplot(fig)
        

        
if __name__ == "__main__":
    OptionPricingApp()

    # Optional plots after calculation, if inputs were submitted
    if st.session_state.get("submitted", False):
        st.markdown("### 📊 Graphs")
        if st.checkbox("Show Price vs Spot Graph"):
            plot_price_vs_spot(
                st.session_state._last_K,
                st.session_state._last_T,
                st.session_state._last_r,
                st.session_state._last_sigma,
                st.session_state._last_option_type,
                st.session_state._last_q
            )

        if st.checkbox("Show Delta & Gamma vs Spot Graph"):
            plot_greeks_vs_spot(
                st.session_state._last_K,
                st.session_state._last_T,
                st.session_state._last_r,
                st.session_state._last_sigma,
                st.session_state._last_option_type,
                st.session_state._last_q
            )

        if st.checkbox("Show Implied Volatility Surface"):
            plot_implied_vol_surface(
                st.session_state._last_S,
                st.session_state._last_r,
                st.session_state._last_option_type,
                st.session_state._last_q
            )
