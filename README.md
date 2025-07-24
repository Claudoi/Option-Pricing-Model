# Option Pricing Model

This project implements a **modular and interactive platform** for pricing European, American, and exotic options and performing advanced portfolio risk analysis. Designed for **educational use, quant research, and professional prototyping**, it includes a Streamlit web app, reusable pricing/risk modules, and detailed notebooks.


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
- 📉 Stress Testing: custom loss estimation under defined shock scenarios
- 📐 Risk Ratios Calculation:
  - Sharpe Ratio, Sortino Ratio, Calmar Ratio, Omega Ratio
  - Information Ratio, Skewness, Kurtosis
  - Max Drawdown, Value at Risk, Expected Shortfall
  - 📊 Includes dynamic bar chart visualization of all ratios
- 🔗 Automatic price data download from Yahoo Finance (no manual CSV uploads)
- 📓 Jupyter notebooks for step-by-step exploration of all pricing and risk models
- 🧪 Unit tests for core pricing and risk analytics functionality
- 🧱 Modular architecture, designed for reusability and expansion
- 🖥️ Streamlit-based web application with user-friendly UI


## 📁 Project Structure

```
option-pricing-model/
│
├── notebooks/ # Step-by-step exploration of models
│ ├── 01_black_scholes.ipynb
│ ├── 02_binomial_american.ipynb
│ ├── 03_monte_carlo_asian.ipynb
│ ├── ... (others: barrier, lookback, greeks, interface)
│
├── src/ # Core pricing logic
│ ├── pricing_black_scholes.py
│ ├── pricing_binomial.py
│ ├── pricing_montecarlo.py
│ ├── greeks.py
│ ├── constants.py
│ ├── risk_analysis.py
│ ├── risk_ratios.py
│ └── utils.py
│
├── tests/ # Unit tests
│ └── greeks_tests.py
│ └── pricing_tests.py
│
├── webapp / # Streamlit web app
├── app.py 
│
├── requirements.txt # Dependencies
├── data/ # Optional data files
│ └── options_sample.csv
└── README.md
```


## Installation

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


## Usage

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


## 🧠 Future Ideas

- Volatility Surface Calibration using market option data (e.g., SVI or SABR models)
- Local and Stochastic Volatility Models such as Heston or Dupire
- Delta Hedging Simulator to evaluate dynamic hedging performance
- Portfolio Optimization with risk-adjusted objective functions (e.g., CVaR, Sharpe)
- Intraday Risk Monitoring with real-time rolling VaR and Expected Shortfall


## License

MIT License


## Author
Developed by Claudio Martel
