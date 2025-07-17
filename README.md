# Option Pricing Model

This project implements a **modular and interactive system** in Python to price European, American, and exotic options. It is designed for educational purposes, quant research, and professional financial prototyping.


## Features

- 📈 **Black-Scholes Model** (closed-form) for European options
- 🌲 **Binomial Tree Model** for European and American options
- 🎲 **Monte Carlo Simulation** for exotic options:
  - Asian (arithmetic and geometric average)
  - Lookback (fixed and floating)
  - Digital barrier (knock-in, knock-out)
- 🧮 **Greeks Calculation**: Delta, Gamma, Vega, Theta, Rho
- ✅ Robust input validation and reusable utilities
- 📊 **Interactive Interface with Streamlit**: explore and visualize models dynamically
- 🧪 **Unit tests** for key pricing components
- 📓 Jupyter notebooks for modular exploration
- 🧱 Clean, extensible architecture ready for future extensions


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

- Heston model, SABR or stochastic volatility
- American options via Longstaff-Schwartz in Monte Carlo
- Calibration to market data (implied volatility surface)
- Portfolio-level VaR/ES


## License

MIT License


## Author
Developed by Claudio Martel