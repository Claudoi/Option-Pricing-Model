# Option Pricing Model

This project implements a modular system in Python for the valuation of European, American, and exotic options. The goal is to provide a well-structured, testable, and extensible codebase that supports both educational use and professional financial analysis.


## Features

- Black-Scholes model for European options
- Binomial trees for American options
- Monte Carlo simulations for exotic options (Asian, barrier, lookback)
- Calculation of option Greeks (Delta, Gamma, Vega, Theta, Rho)
- Interactive visualizations using Matplotlib and Plotly
- Jupyter notebooks for step-by-step exploration
- Streamlit web app prototype for user interaction
- Unit testing for pricing functions


## Project Structure

option-pricing-model/
│
├── notebooks/ # Jupyter notebooks for each model
│ ├── 01_black_scholes.ipynb
│ ├── 02_binomial_american.ipynb
│ ├── ...
│
├── src/ # Core pricing functions and helpers
│ ├── pricing_black_scholes.py
│ ├── pricing_binomial.py
│ ├── pricing_montecarlo.py
│ ├── greeks.py
│ ├── utils.py
│
├── tests/ # Unit tests
│ └── test_pricing.py
│
├── webapp/ # Streamlit app
│ └── app.py
│
├── data/ # Optional data files
│ └── options_sample.csv
│
├── requirements.txt # Dependencies
├── .gitignore # Git ignore rules
└── README.md # Project documentation


## Installation

Clone the repository and install dependencies:

```bash
git clone https://github.com/your-username/option-pricing-model.git
cd option-pricing-model
pip install -r requirements.txt
```

Or create a virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Usage

You can:

- Run and explore the models in the Jupyter notebooks under notebooks/

- Import pricing functions from src/ into other projects

- Launch the interactive app (if developed) from webapp/app.py

- Extend the codebase to support additional derivatives

Example usage in Python:

```python
from src.pricing_black_scholes import black_scholes

price = black_scholes(S=100, K=100, T=1, r=0.05, sigma=0.2, option_type='call')
print(f"Call option price: {price:.2f}")
```


## Tests

Run unit tests using:

```bash
pytest tests/
```


## License

MIT License


## Author
Developed by Claudio Martel Flores