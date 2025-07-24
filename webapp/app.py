import streamlit as st, numpy as np, matplotlib.pyplot as plt, pandas as pd, plotly.graph_objects as go
import sys, os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.pricing_black_scholes import BlackScholesOption
from src.pricing_montecarlo import MonteCarloOption
from src.pricing_binomial import BinomialOption
from src.greeks import BlackScholesGreeks
from src.risk_analysis import  PortfolioVaR, HistoricalVaR, MonteCarloVaR, RollingVaR, StressTester
from src.volatility_surface import VolatilitySurface
from src.utils import fetch_returns_from_yahoo


class OptionPricingApp:
    def __init__(self):
        self.setup_ui()

    def setup_ui(self):
        st.title("\U0001F4C8 Option Pricing Interface")
        st.markdown("""
            Explore option pricing models and portfolio risk tools:

            - Black-Scholes, Binomial & Monte Carlo (exotic & American options)
            - Greeks & implied volatility surfaces
            - Value at Risk (Parametric, Historical, Monte Carlo)
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
        st.session_state.selected_model = self.model

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
                self.risk_method = st.selectbox(
                    "Risk method",
                    ["Parametric", "Historical", "Monte Carlo", "Rolling VaR (EWMA)", "Rolling VaR (GARCH)", "Stress Testing"]
                )
                self.confidence_level = st.slider("Confidence Level", 0.80, 0.99, 0.95, step=0.01)
                self.holding_period = st.number_input("Holding period (days)", min_value=1, value=1)
                self.tickers = st.text_input("Enter asset tickers (comma-separated)", value="NVDA")
                self.start_date = st.date_input("Start date")
                self.end_date = st.date_input("End date")

                if self.risk_method == "Monte Carlo":
                    self.n_sim = st.slider("Number of simulations", 1000, 50000, 10000, step=1000)

                elif self.risk_method == "Rolling VaR (EWMA)":
                    self.lambda_ = st.slider("Lambda (EWMA decay)", 0.80, 0.99, 0.94, step=0.01)
                    self.window = st.number_input("Window size", min_value=30, value=100)

                elif self.risk_method == "Rolling VaR (GARCH)":
                    self.window = st.number_input("Window size", min_value=30, value=100)


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

                    df_returns = fetch_returns_from_yahoo(tickers_list, str(self.start_date), str(self.end_date))
                    returns = df_returns.values
                    assets = df_returns.columns
                    n_assets = len(assets)

                    # Equal weights if no other information is provided
                    weights = np.ones(n_assets) / n_assets

                    if self.risk_method == "Parametric":
                        model = PortfolioVaR(pd.DataFrame(returns, columns=assets), weights,
                                            confidence_level=self.confidence_level,
                                            holding_period=self.holding_period)
                        var = model.calculate_var()
                        es = model.calculate_es()
                        st.success(f"Parametric VaR: {var:.4f}")
                        st.info(f"Expected Shortfall (ES): {es:.4f}")


                    elif self.risk_method == "Historical":
                        portfolio_returns = returns @ weights
                        model = HistoricalVaR(portfolio_returns, confidence_level=self.confidence_level)
                        var = model.calculate_var()
                        es = model.calculate_es()
                        st.success(f"Historical VaR: {var:.4f}")
                        st.info(f"Expected Shortfall (ES): {es:.4f}")

                        # Plotly histogram
                        fig = go.Figure()
                        fig.add_trace(go.Histogram(x=portfolio_returns, nbinsx=50, name="Returns", marker_color="skyblue"))
                        fig.add_vline(x=-var, line=dict(color="red", dash="dash"), name="VaR")
                        fig.add_vline(x=-es, line=dict(color="orange", dash="dash"), name="ES")
                        fig.update_layout(
                            title="Historical Portfolio Return Distribution",
                            xaxis_title="Return",
                            yaxis_title="Frequency",
                            template="plotly_dark"
                        )
                        st.plotly_chart(fig, use_container_width=True)


                    elif self.risk_method == "Monte Carlo":
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

                        # Plotly histogram
                        fig = go.Figure()
                        fig.add_trace(go.Histogram(x=simulated_returns, nbinsx=50, name="Simulated Returns", marker_color="skyblue"))
                        fig.add_vline(x=-var, line=dict(color="red", dash="dash"), name="VaR")
                        fig.add_vline(x=-es, line=dict(color="orange", dash="dash"), name="ES")
                        fig.update_layout(
                            title="Monte Carlo Simulated Portfolio Returns",
                            xaxis_title="Return",
                            yaxis_title="Frequency",
                            template="plotly_dark"
                        )
                        st.plotly_chart(fig, use_container_width=True)


                    elif self.risk_method.startswith("Rolling VaR"):

                        portfolio_returns = returns @ weights
                        method = "ewma" if "EWMA" in self.risk_method else "garch"
                        lambda_ = getattr(self, "lambda_", 0.94)

                        if len(portfolio_returns) < self.window:
                            st.warning("⚠️ Not enough data to compute Rolling VaR. Increase the date range or reduce the window size.")
                            return

                        try:
                            model = RollingVaR(
                                returns=portfolio_returns,
                                method=method,
                                lambda_=lambda_,
                                window=self.window,
                                confidence_level=self.confidence_level
                            )
                            var_series = model.calculate_var_series()

                            if method == "ewma":
                                st.success(f"EWMA Rolling VaR (last value): {var_series[-1]:.4f}")
                            else:
                                st.success(f"GARCH VaR (1-day ahead): {var_series:.4f}")

                            # Plot only if EWMA
                            if method == "ewma":
                                fig = go.Figure()
                                fig.add_trace(go.Scatter(y=var_series, mode="lines", name="Rolling VaR (EWMA)"))
                                fig.update_layout(
                                    title="Rolling Value at Risk",
                                    xaxis_title="Time",
                                    yaxis_title="VaR",
                                    template="plotly_dark"
                                )
                                st.plotly_chart(fig, use_container_width=True)

                            else:
                                    fig = go.Figure()
                                    fig.add_trace(go.Bar(y=[var_series], name="GARCH VaR (1-day ahead)"))
                                    fig.update_layout(
                                        title="GARCH VaR (1-day ahead forecast)",
                                        yaxis_title="VaR",
                                        template="plotly_dark"
                                    )
                                    st.plotly_chart(fig, use_container_width=True)

                        except Exception as e:
                            st.warning("⚠️ Rolling VaR calculation failed.")
                            st.code(str(e))


                    elif self.risk_method == "Stress Testing":
                        portfolio_returns = returns @ weights
                        base_value = 1_000_000  # Valor del portafolio, puedes hacerlo input

                        # Define escenarios de stress
                        scenarios = {
                            "Mild Shock (-2%)": -0.02,
                            "Moderate Shock (-5%)": -0.05,
                            "Severe Shock (-10%)": -0.10,
                            "Extreme Shock (-20%)": -0.20,
                        }

                        st.subheader("📉 Stress Scenarios")

                        stress_results = {}
                        for name, shock in scenarios.items():
                            shocked_returns = portfolio_returns + shock  # Simulación simple
                            new_value = base_value * (1 + shocked_returns.mean())
                            loss = base_value - new_value
                            stress_results[name] = loss

                        # Mostrar resultados
                        for scenario, loss in stress_results.items():
                            st.info(f"{scenario}: Estimated Loss = ${loss:,.2f}")

                        # Plotly bar chart
                        fig = go.Figure()
                        fig.add_trace(go.Bar(
                            x=list(stress_results.keys()),
                            y=list(stress_results.values()),
                            name="Loss",
                            marker_color="crimson"
                        ))
                        fig.update_layout(
                            title="Stress Testing Results",
                            xaxis_title="Scenario",
                            yaxis_title="Estimated Portfolio Loss",
                            template="plotly_dark"
                        )
                        st.plotly_chart(fig, use_container_width=True)



                except Exception as e:
                    st.error("⚠️ Error during risk analysis.")
                    st.code(str(e))
                    st.info("🔎 Please check:\n- That all tickers are valid.\n- That the date range includes trading days.\n- That the API returned price data.")


        except Exception as e:
            st.error(f"Error during calculation: {str(e)}")



def plot_price_vs_spot(K, T, r, sigma, option_type, q):
    S_range = np.linspace(50, 150, 100)
    prices = [BlackScholesOption(S, K, T, r, sigma, option_type, q).price() for S in S_range]

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=S_range, y=prices, mode='lines', name='Price'))
    fig.update_layout(
        title="Option Price vs Spot Price",
        xaxis_title="Spot Price (S)",
        yaxis_title="Option Price",
        template="plotly_dark"
    )
    st.plotly_chart(fig, use_container_width=True)


def plot_greeks_vs_spot(K, T, r, sigma, option_type, q):
    S_range = np.linspace(50, 150, 100)
    deltas, gammas = [], []

    for S in S_range:
        opt = BlackScholesOption(S, K, T, r, sigma, option_type, q)
        greeks = opt.greeks()
        deltas.append(greeks["delta"])
        gammas.append(greeks["gamma"])

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=S_range, y=deltas, mode='lines', name='Delta'))
    fig.add_trace(go.Scatter(x=S_range, y=gammas, mode='lines', name='Gamma'))
    fig.update_layout(
        title="Greeks vs Spot Price",
        xaxis_title="Spot Price (S)",
        yaxis_title="Value",
        template="plotly_dark"
    )
    st.plotly_chart(fig, use_container_width=True)



def plot_market_vol_surface(ticker: str, interpolation_method: str = "linear"):
    st.markdown(f"### 📉 Calibrated Implied Volatility Surface for {ticker.upper()}")

    try:
        calibrator = VolatilitySurface(ticker)
        calibrator.fetch_data()

        if calibrator.IV is None or len(calibrator.IV) == 0:
            st.warning("⚠️ No implied volatility data found.")
            return

        grid_K, grid_T, grid_IV = calibrator.interpolate(method=interpolation_method)

        fig = go.Figure(data=[go.Surface(x=grid_K, y=grid_T, z=grid_IV)])
        fig.update_layout(
            scene=dict(
                xaxis_title="Strike (K)",
                yaxis_title="Maturity (T)",
                zaxis_title="Implied Volatility"
            ),
            template="plotly_dark",
            margin=dict(l=0, r=0, t=30, b=0)
        )
        st.plotly_chart(fig, use_container_width=True)

    except Exception as e:
        st.error("⚠️ Error during volatility surface calibration.")
        st.code(str(e))



if __name__ == "__main__":
    OptionPricingApp()

    if st.session_state.get("submitted", False) and st.session_state.get("selected_model") == "Black-Scholes":
        st.markdown("### 📊 Graphs (Black-Scholes)")

        if st.checkbox("Price vs Spot Graph"):
            plot_price_vs_spot(
                st.session_state._last_K,
                st.session_state._last_T,
                st.session_state._last_r,
                st.session_state._last_sigma,
                st.session_state._last_option_type,
                st.session_state._last_q
            )

        if st.checkbox("Delta & Gamma vs Spot Graph"):
            plot_greeks_vs_spot(
                st.session_state._last_K,
                st.session_state._last_T,
                st.session_state._last_r,
                st.session_state._last_sigma,
                st.session_state._last_option_type,
                st.session_state._last_q
            )

        if st.checkbox("Market Implied Volatility Surface"):
            ticker = st.text_input("Enter ticker symbol for IV surface", value="NVDA")
            interpolation_method = st.selectbox("Interpolation method", ["linear", "cubic", "nearest"], key="interp_method")

            if st.button("Generate Surface"):
                plot_market_vol_surface(ticker, interpolation_method)
