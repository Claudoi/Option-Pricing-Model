# Option Pricing Model

[![Python](https://img.shields.io/badge/Python-3.8+-blue?logo=python&logoColor=white)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-green?logo=Open%20Source%20Initiative&logoColor=white)](https://opensource.org/licenses/MIT)
[![Last Commit](https://img.shields.io/badge/Last_Commit-Recently-brightgreen?logo=git&logoColor=white)](https://github.com/Claudoi/option-pricing-model/commits/main)
[![Streamlit](https://img.shields.io/badge/Open%20in%20Streamlit-App-red?logo=streamlit&logoColor=white)](https://options-pricing-models.streamlit.app/)


This project implements a **modular and interactive platform** for pricing European, American, and exotic options and performing advanced portfolio risk analysis. Designed for **educational use, quant research, and professional prototyping**, it includes a Streamlit web app, reusable pricing/risk modules, and detailed notebooks.

🚀 **Live App**: [options-pricing-models.streamlit.app](https://options-pricing-models.streamlit.app/)


## Features

- 📈 Black-Scholes Model (closed-form) for European options
- 🌲 Binomial Tree Model for European and American options
- 🎲 Monte Carlo Simulation for exotic and American options:
  - Asian (arithmetic and geometric average)
  - Lookback (fixed and floating)
  - Digital barrier options (knock-in, knock-out)
  - American options via Longstaff-Schwartz (LSM)
- 🧮 Greeks Calculation: Delta, Gamma, Vega, Theta, Rho
- 📊 Risk Analysis with VaR and Expected Shortfall (ES):
  - Parametric (variance-covariance)
  - Historical
  - Monte Carlo simulation-based
  - Rolling VaR with:
    - EWMA (Exponentially Weighted Moving Average)
    - GARCH(1,1)
- 📐 Risk Ratios Calculation:
  - Sharpe Ratio, Sortino Ratio, Calmar Ratio, Omega Ratio
  - Information Ratio, Skewness, Kurtosis
  - Max Drawdown, Value at Risk, Expected Shortfall
  - Custom scenario-based stress testing
  - 📊 Includes dynamic bar chart visualization of all ratios
- 📉 Volatility Surface Calibration
  - SVI (Stochastic Volatility Inspired) model
  - SABR (Stochastic Alpha Beta Rho) model
  - Dupire Local Volatility (via PDE)
  - Heston Stochastic Volatility Model
- 🧭 Delta Hedging Simulator
  - Interactive simulator for delta hedging under Black-Scholes and Heston models
  - PnL decomposition: Delta, Theta, Residual
- 🔗 Automatic price data download from Yahoo Finance and also Manual CSV 
- 📓 Jupyter notebooks for step-by-step exploration of all pricing and risk models
- 🧪 Unit tests for core pricing and risk analytics functionality
- 🧱 Modular architecture, designed for reusability and expansion
- 🖥️ Streamlit-based web application with user-friendly UI


## 📁 Project Structure

```
option-pricing-model/
│
├── data/
│   ├── options_sample.csv
│   └── tempus_options_comparison.csv
│
├── notebooks/
│   ├── 01_black_scholes.ipynb
│   ├── 02_binomial_american.ipynb
│   ├── 03_monte_carlo_asian.ipynb
│   ├── 04_digital_barrier.ipynb
│   ├── 05_lookback_options.ipynb
│   ├── 06_greeks_analysis.ipynb
│   ├── 07_visualizations.ipynb
│   └── 08_interface_streamlit.ipynb
│
├── src/
│   ├── hedging/
│   │   ├── hedging_simulator.py
│   │   ├── heston_hedging_simulator.py
│   │   └── hedging_pnl_decomposition.py
│   │
│   ├── models/
│   │   ├── pricing_black_scholes.py
│   │   ├── pricing_binomial.py
│   │   ├── pricing_montecarlo.py
│   │   ├── greeks.py
│   │   └── implied_volatility.py
│   │
│   ├── risk/
│   │   ├── risk_analysis.py
│   │   ├── risk_ratios.py
│   │   └── risk_rolling.py
│   │
│   ├── utils/
│   │   ├── constants.py
│   │   ├── utils.py
│   │   └── plot_utils.py
│   │
│   └── volatility/
│       ├── volatility_surface.py
│       ├── svi_calibration.py
│       ├── sabr_calibration.py
│       ├── local_volatility.py
│       └── stochastic_volatility.py
│
├── tests/
│   ├── greeks_tests.ipynb
│   └── pricing_tests.ipynb
│
├── webapp/
│   └── app.py
│
├── requirements.txt
├── LICENSE
└── README.md

```


## ⚙️ Installation

Clone the repository and install dependencies:

```bash
git clone https://github.com/Claudoi/option-pricing-model.git
cd option-pricing-model
pip install -r requirements.txt
```

Or create a virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```


## 💻 Usage

### 1 - Run the app interactively

```bash
streamlit run app.py
```

### 2 - Use notebooks for step-by-step exploration

Open any notebook from /notebooks/ to study each pricing model in depth.

### 3 - Use the pricing functions in your own code

```python
from src.pricing_black_scholes import BlackScholesOption

bs = BlackScholesOption(S=100, K=100, T=1, r=0.05, sigma=0.2, option_type="call")
price = bs.price()
print(f"Call price: {price:.2f}")
```

### 4 - Run unit tests

```bash
pytest tests/
```


## 🧠 Next Module: Machine Learning for Volatility Forecasting
- Build deep learning models (MLP, LSTM) to forecast implied volatility surfaces
- Train on historical option prices, strike/maturity grids, market conditions
- Evaluate forecast error and hedging performance


## 📄 License

MIT License


## 👨‍💻 Author
Developed by Claudio Martel
