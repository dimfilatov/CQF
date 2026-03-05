## Overview

This repository contains assignments for CQF Module 1 covering market analytics, binomial option pricing, and stochastic simulations useful for derivative pricing and risk analysis.

---

## Assignment 1 — Market Analytics

Provides tools for loading, processing, and analyzing financial time series in `market_analytics.py`.

- **Purpose**: data ingestion, feature engineering (lags), returns computation (simple and log), volatility statistics, and visual diagnostics (Q-Q plots, distribution plots).
- **Main entry point**: `MarketAnalytics` class in `market_analytics.py`.
- **Usage**:

		- Command line: `python market_analytics.py`
		- As a module:

				from market_analytics import MarketAnalytics
				analytics = MarketAnalytics(ticker='^GSPC', period='max')
				analytics.main()

- **Dependencies**: `pandas`, `numpy`, `matplotlib`, `scipy`, plus a data source helper (e.g., `quantmod.markets` or similar).

---

## Assignment 2 — Binomial Option Pricing Model

Implements the Cox-Ross-Rubinstein binomial tree for pricing European and American options.

- **Purpose**: Construct n-step binomial trees, compute terminal payoffs, perform backward induction under a risk-neutral measure, and compute hedging ratios (delta).
- **Files**: `binomial_option_tree.py`, `binomial_option_tree.ipynb`.
- **Features**: support for calls/puts, European/American styles, early exercise checks, and Greeks calculation.

---

## Assignment 3 — Stochastic Simulations

Adds `price_simulator.py`, a notebook-friendly simulation helper for stochastic price models.

- **Key features**:
	- GBM (Geometric Brownian Motion): exact (log-normal) sampling and Euler–Maruyama discretization.
	- Ornstein–Uhlenbeck (OU): exact discrete update and Euler option for mean-reverting dynamics.
	- Correlated multi-asset Euler simulation using a Cholesky-decomposed correlation matrix.
	- Plotting helpers for quick matplotlib visuals and optional Plotly conversion.

- **File**: `price_simulator.py`

- **Quick notebook usage**:

```python
from price_simulator import PriceSimulator
sim = PriceSimulator(S0=100, mu=0.05, sigma=0.2, seed=42)
paths_exact = sim.simulate_gbm(steps=252, paths=10, method='exact')
sim.plot(paths_exact, title='GBM (exact) — 10 paths')
```

---

**Module**: CQF Module 1  
**Topic**: Derivative Pricing Fundamentals
