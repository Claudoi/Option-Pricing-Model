import streamlit as st
import numpy as np
import pandas as pd

from src.risk.risk_analysis import (
    PortfolioVaR, HistoricalVaR, MonteCarloVaR
)
from src.risk.risk_rolling import RollingVaR
from src.risk.risk_ratios import RiskRatios
from src.utils.utils import fetch_returns_from_yahoo
from src.utils.plot_utils import PlotUtils


def risk_ui():
    # Header
    st.markdown("## Portfolio Risk Analysis & VaR/ES")
    st.markdown('<div class="small-muted">Download historical data • Compute ratios • Visualize VaR & ES</div>', unsafe_allow_html=True)

    # Input Form
    st.markdown('<div class="card" style="padding:1rem;">', unsafe_allow_html=True)
    with st.form("risk_data_form"):
        # Portfolio tickers
        tickers_str = st.text_input("Portfolio Tickers (comma separated)", "AAPL, MSFT")
        tickers = [t.strip().upper() for t in tickers_str.split(",") if t.strip()]

        # Dates
        c1, c2 = st.columns(2)
        with c1:
            start = st.date_input("Start Date", value=pd.to_datetime("2021-01-01"))
        with c2:
            end = st.date_input("End Date", value=pd.to_datetime("today"))

        # Weights
        weights_str = st.text_input("Portfolio Weights (comma separated, must sum 1)", "0.5, 0.5")
        try:
            weights = np.array([float(w) for w in weights_str.split(",")])
        except ValueError:
            st.error("Weights must be numeric and comma-separated.")
            st.stop()

        # Method & parameters
        method = st.selectbox("Select VaR Method", ["Parametric", "Historical", "Monte Carlo", "Rolling EWMA", "Rolling GARCH"])
        confidence = st.slider("Confidence Level", 0.90, 0.99, 0.95, step=0.01)
        window = st.number_input("Rolling Window (for EWMA/GARCH)", value=100, min_value=10, step=1)
        lambda_ = st.number_input("EWMA Lambda", value=0.94, min_value=0.7, max_value=0.99, step=0.01)

        submitted = st.form_submit_button("Load & Analyze")
    st.markdown('</div>', unsafe_allow_html=True)

    if not submitted:
        st.stop()

    try:
        # Data Download
        returns = fetch_returns_from_yahoo(tickers, str(start), str(end))
        if returns.empty:
            st.error("No data returned for the given tickers and date range.")
            st.stop()

        portfolio_returns = returns @ weights

        # Risk Ratios
        risk_ratios = RiskRatios(portfolio_returns)
        ratios_dict = {
            "Sharpe": risk_ratios.sharpe_ratio(),
            "Sortino": risk_ratios.sortino_ratio(),
            "Calmar": risk_ratios.calmar_ratio(),
            "Max Drawdown": risk_ratios.max_drawdown(),
            "Omega": risk_ratios.omega_ratio(),
            "Skewness": risk_ratios.skewness(),
            "Kurtosis": risk_ratios.kurtosis(),
            f"VaR {int(confidence*100)}%": risk_ratios.value_at_risk(1-confidence),
            f"ES {int(confidence*100)}%": risk_ratios.expected_shortfall(1-confidence),
        }

        st.markdown('<div class="card" style="padding:1rem;">', unsafe_allow_html=True)
        st.markdown("#### Main Risk Ratios")
        cols = st.columns(len(ratios_dict))
        for col, (k, v) in zip(cols, ratios_dict.items()):
            col.metric(label=k, value=f"{v:.5f}" if isinstance(v, float) else v)
        st.markdown('</div>', unsafe_allow_html=True)

        # VAR / ES Analysis
        var, es, var_series = None, None, None
        st.markdown('<div class="card" style="padding:1rem;">', unsafe_allow_html=True)
        st.markdown(f"#### {method} VaR Analysis")

        if method == "Parametric":
            var_model = PortfolioVaR(returns, weights, confidence_level=confidence)
            var = var_model.calculate_var()
            es = var_model.calculate_es()
            st.success(f"Parametric VaR: {var:.4%} | ES: {es:.4%}")

        elif method == "Historical":
            var_model = HistoricalVaR(portfolio_returns, confidence_level=confidence)
            var = var_model.calculate_var()
            es = var_model.calculate_es()
            st.success(f"Historical VaR: {var:.4%} | ES: {es:.4%}")

        elif method == "Monte Carlo":
            mu = returns.mean().values
            sigma = returns.std().values
            var_model = MonteCarloVaR(returns.iloc[-1].values, mu, sigma, weights, confidence_level=confidence)
            var, es, _ = var_model.calculate_var_es()
            st.success(f"Monte Carlo VaR: {var:.4%} | ES: {es:.4%}")

        elif method == "Rolling EWMA":
            rolling = RollingVaR(portfolio_returns, method="ewma", confidence_level=confidence, window=window, lambda_=lambda_)
            var_series = rolling.calculate_var_series()
            st.plotly_chart(
                PlotUtils.plot_rolling_var(portfolio_returns, var_series, method="ewma", confidence_level=confidence, window=window),
                use_container_width=True
            )
            st.info("Rolling VaR curve (EWMA).")
            var = var_series[-1] if var_series is not None else None

        elif method == "Rolling GARCH":
            try:
                rolling = RollingVaR(portfolio_returns, method="garch", confidence_level=confidence, window=window)
                var_series = rolling.calculate_var_series()
                st.plotly_chart(PlotUtils.plot_garch_var_bar(var_series), use_container_width=True)
                var = var_series[-1] if var_series is not None else None
            except ImportError:
                st.error("You need to install the 'arch' package for GARCH.")

        # Histogram with VaR/ES
        if var is not None and es is not None:
            st.plotly_chart(
                PlotUtils.plot_var_es_histogram(
                    portfolio_returns, var, es,
                    title=f"Return Distribution with VaR/ES ({int(confidence*100)}%)"
                ),
                use_container_width=True
            )
        st.markdown('</div>', unsafe_allow_html=True)

    except Exception as e:
        st.error(f"Error loading data or calculating risk metrics: {e}")
