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

## Assignment 3 — Stochastic Process Simulations

Implements `stochastic_process_simulator.py`, a comprehensive library for simulating price and rate models commonly used in derivative pricing.

- **Key features**:
	- **GBM (Geometric Brownian Motion)**: exact (log-normal) sampling and Euler–Maruyama discretization.
	- **Ornstein–Uhlenbeck (OU)**: mean-reverting dynamics with Euler-Maruyama approximation.
	- **Cox-Ingersoll-Ross (CIR)**: square-root process for short rate modeling with Euler-Maruyama approximation.
	- **Correlated multi-asset Euler simulation**: using Cholesky-decomposed correlation matrix for modeling dependent assets.
	- **Plotting helpers**: quick matplotlib visualization for simulation paths.

- **File**: `stochastic_process_simulator.py`

- **Main class**: `StochasticProcessSimulator`

- **Quick notebook usage**:

```python
from stochastic_process_simulator import StochasticProcessSimulator
sim = StochasticProcessSimulator(S0=100, mu=0.05, sigma=0.2, seed=42)
paths_exact = sim.simulate_gbm(steps=252, paths=10, method='exact')
sim.plot(paths_exact, title='GBM (exact) — 10 paths')

# OU process example
paths_ou = sim.simulate_ou(steps=252, paths=10, theta=1.0, mu_level=0.05, sigma=0.03)
sim.plot(paths_ou, title='OU process — 10 paths')

# Correlated assets example
paths_corr = sim.simulate_correlated_euler(steps=252, n_assets=2, n_paths=5, mu=[0.05, 0.03], sigma=[0.2, 0.1], corr=[[1.0, 0.8], [0.8, 1.0]], S0=[100, 50])
sim.plot(paths_corr[:, 0, :], title='Correlated GBM (Asset 1)')
```

---

**Module**: CQF Module 1  
**Topic**: Derivative Pricing Fundamentals
