# CQF Module 1 - Binomial Option Pricing Model

## Overview

This module contains the first assignment for the Certificate in Quantitative Finance (CQF) program, focusing on the **Binomial Option Pricing Model** - a foundational method for valuing derivatives.

### Assignment 1: Market Analytics

This addition contains tools for loading, processing, and analyzing financial time series in `market_analytics.py`.

- **What it does**: Loads market data (via `quantmod.markets.getData`), creates lagged features, computes simple and log returns, calculates adjusted volatility statistics, compares even/odd day return behavior, and produces Q-Q and distribution plots for visual analysis.
- **Main entry point**: the `MarketAnalytics` class. Example usage:

	- Command line: `python market_analytics.py` (runs the example at the bottom of the script)
	- As a module:

		from market_analytics import MarketAnalytics
		analytics = MarketAnalytics(ticker='^GSPC', period='max')
		analytics.main()

- **Dependencies**: `pandas`, `numpy`, `matplotlib`, `scipy`, and the `quantmod` package or local `quantmod.markets` module used for data retrieval.
- **Notes**: Ensure required packages are installed and that the environment has network access for fetching data from Yahoo Finance or the configured data source.

## Assignment 2: Binomial Option Pricing Model

### Introduction

The Binomial Tree Model, introduced by Cox, Ross, and Rubinstein in 1979, is a discrete-time framework for modeling the evolution of an asset's price over time. It is particularly useful for valuing both European and American-style options due to its flexibility and intuitive structure.

### Key Concepts

#### Model Framework

The binomial model assumes that at each time step, the underlying asset price can move in one of two directions:
- **Up-move**: Price increases by a factor $u$
- **Down-move**: Price decreases by a factor $v$

Starting from an initial price $S_0$, after one time step the asset can reach:
- $S_0 \cdot u$ (up-move)
- $S_0 \cdot v$ (down-move)

After multiple steps, this creates a tree-like structure representing all possible future price paths.

#### Risk-Neutral Probability

The model employs a risk-neutral measure where each share price today is the discounted expectation of future prices. The key formulas are:

**Up-move factor:**
$$u = 1 + \sigma\sqrt{\Delta t}$$

**Down-move factor:**
$$v = 1 - \sigma\sqrt{\Delta t}$$

where $\sigma$ is volatility and $\Delta t$ is the time step size.

**Risk-neutral probability:**
$$p' = \frac{1}{2} + \frac{r\sqrt{\Delta t}}{2\sigma}$$

where $p'$ represents the probability of an up-move under the risk-neutral measure.

**Option value at each node:**
$$V = \frac{1}{1 + r\Delta t}(p' \cdot V^{+} + (1-p') \cdot V^{-})$$

where $V$ is the discounted present value of expected future option payoffs, $V^{+}$ is the up value, and $V^{-}$ is the down value.

### Building the Binomial Tree

The construction of a binomial tree involves a systematic 5-step process:

1. **Step 1**: Draw an n-step tree structure representing all possible price paths
2. **Step 2**: Calculate terminal asset prices at the end of n steps
3. **Step 3**: Determine option payoff values at each terminal node based on the option type (call/put)
4. **Step 4**: Discount values backward from step n to n-1 using risk-neutral probabilities
5. **Step 5**: Repeat the discounting process until reaching the initial option value at step 0

### Model Capabilities

The binomial tree model supports:
- **Option Types**: European and American options (calls and puts)
- **Early Exercise**: American options can be evaluated for optimal early exercise at each node
- **Greeks Calculation**: Computes hedging ratios (delta) for risk management
- **Flexibility**: Accommodates various market conditions including time to maturity, volatility, and interest rates

### Implementation Details

The assignment includes a Python implementation with a `binomial_tree` class that:
- Parameterizes the model with steps, maturity time, risk-free rate, volatility, spot price, and strike price
- Constructs the price tree through forward iteration
- Calculates option values through backward induction
- Computes Greeks for hedging strategies
- Supports both European and American style options

### Files in This Assignment

- `binomial_option_tree.ipynb`: Jupyter notebook with detailed explanations, formulas, and implementation
- `binomial_option_tree.py`: Python script containing the core binomial tree implementation
- `market_analytics.py`: Python module providing market data loading, returns/volatility analysis, and plotting utilities
- `README.md`: This documentation file

### Key Takeaways

The binomial model provides:
✓ An intuitive discrete-time framework for option pricing
✓ Flexibility to handle American options with early exercise features
✓ Computational efficiency with backward induction
✓ Foundation for understanding continuous-time models
✓ Practical Greeks calculations for risk management

---

**Module**: CQF Module 1  
**Topic**: Derivative Pricing Fundamentals  
**Method**: Binomial Option Pricing Model

---