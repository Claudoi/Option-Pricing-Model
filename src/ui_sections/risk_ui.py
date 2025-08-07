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
    st.markdown("### 📊 Portfolio Risk Analysis & VaR/ES")

    with st.form("risk_data_form"):
        st.markdown("#### Portfolio Settings")
        tickers_str = st.text_input("Portfolio Tickers (comma separated)", "AAPL, MSFT")
        tickers = [t.strip().upper() for t in tickers_str.split(",") if t.strip()]

        col1, col2 = st.columns(2)
        with col1:
            start = st.date_input("Start Date", value=pd.to_datetime("2021-01-01"))
        with col2:
            end = st.date_input("End Date", value=pd.to_datetime("today"))

        weights_str = st.text_input("Portfolio Weights (comma separated, must sum 1)", "0.5, 0.5")
        weights = np.array([float(w) for w in weights_str.split(",")])
        method = st.selectbox("Select VaR Method", ["Parametric", "Historical", "Monte Carlo", "Rolling EWMA", "Rolling GARCH"])
        confidence = st.slider("Confidence Level", 0.90, 0.99, 0.95, step=0.01)
        window = st.number_input("Rolling Window (EWMA/GARCH)", value=100, min_value=10, step=1)
        lambda_ = st.number_input("EWMA Lambda", value=0.94, min_value=0.7, max_value=0.99, step=0.01)

        submitted = st.form_submit_button("Load & Analyze")

    if submitted:
        try:
            # 1. Fetch returns
            returns = fetch_returns_from_yahoo(tickers, str(start), str(end))
            st.write("#### Sample of returns", returns.head())

            # 2. Portfolio returns
            portfolio_returns = returns @ weights

            # --- Risk Ratios ---
            st.markdown("#### 🧮 Main Risk Ratios")
            ratios = RiskRatios(portfolio_returns)
            results = {
                "Sharpe": ratios.sharpe_ratio(),
                "Sortino": ratios.sortino_ratio(),
                "Calmar": ratios.calmar_ratio(),
                "Max Drawdown": ratios.max_drawdown(),
                "Omega": ratios.omega_ratio(),
                "Skewness": ratios.skewness(),
                "Kurtosis": ratios.kurtosis(),
                "VaR 95%": ratios.value_at_risk(0.05),
                "ES 95%": ratios.expected_shortfall(0.05),
            }
            cols = st.columns(len(results))
            for col, (k, v) in zip(cols, results.items()):
                col.metric(label=k, value=f"{v:.5f}" if isinstance(v, float) else v)

            # --- VaR & ES ---
            var, es, var_series = None, None, None

            if method == "Parametric":
                model = PortfolioVaR(returns, weights, confidence)
                var = model.calculate_var()
                es = model.calculate_es()
                st.success(f"📌 Parametric VaR: {var:.4%} | ES: {es:.4%}")

            elif method == "Historical":
                model = HistoricalVaR(portfolio_returns, confidence)
                var = model.calculate_var()
                es = model.calculate_es()
                st.success(f"📌 Historical VaR: {var:.4%} | ES: {es:.4%}")

            elif method == "Monte Carlo":
                mu = returns.mean().values
                sigma = returns.std().values
                model = MonteCarloVaR(returns.iloc[-1].values, mu, sigma, weights, confidence)
                var, es, _ = model.calculate_var_es()
                st.success(f"📌 Monte Carlo VaR: {var:.4%} | ES: {es:.4%}")

            elif method == "Rolling EWMA":
                rolling = RollingVaR(portfolio_returns, method="ewma", confidence_level=confidence, window=window, lambda_=lambda_)
                var_series = rolling.calculate_var_series()
                fig = PlotUtils.plot_rolling_var(portfolio_returns, var_series, method="ewma", confidence_level=confidence, window=window)
                st.plotly_chart(fig, use_container_width=True)
                st.info("Rolling VaR (EWMA shown)")
                var = var_series[-1] if var_series is not None else None

            elif method == "Rolling GARCH":
                try:
                    rolling = RollingVaR(portfolio_returns, method="garch", confidence_level=confidence, window=window)
                    var_series = rolling.calculate_var_series()
                    fig = PlotUtils.plot_garch_var_bar(var_series)
                    st.plotly_chart(fig, use_container_width=True)
                    var = var_series[-1] if var_series is not None else None
                except ImportError:
                    st.error("📦 Please install the `arch` package to use GARCH models.")

            # --- Histogram with VaR/ES ---
            if var is not None and es is not None:
                fig = PlotUtils.plot_var_es_histogram(
                    portfolio_returns, var, es,
                    title=f"#### Return Distribution with VaR/ES ({int(confidence*100)}%)"
                )
                st.plotly_chart(fig, use_container_width=True)

        except Exception as e:
            st.error(f"❌ Error loading data or calculating risk metrics: {e}")
