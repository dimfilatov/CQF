# CQF Coursework Repository

This repository contains coursework and assignments for the Certificate in Quantitative Finance (CQF) program, focusing on financial modeling, derivatives pricing, and portfolio optimization.

## Module 1: Financial Analytics and Derivatives Pricing

This module covers fundamental tools for financial data analysis and derivative pricing models.

### Key Components:
- **Market Analytics**: Tools for loading, processing, and analyzing financial time series data, including returns computation, volatility statistics, and visual diagnostics.
- **Binomial Option Pricing**: Implementation of the Cox-Ross-Rubinstein binomial tree model for pricing European and American options, with support for Greeks calculation.
- **Stochastic Simulations**: Simulation helpers for Geometric Brownian Motion (GBM) and Ornstein-Uhlenbeck processes, including correlated multi-asset simulations.

### Files:
- `binomial_option_tree.py`
- `market_analytics.py`
- `price_simulator.py`

## Module 2: Portfolio Theory and Optimization

This module demonstrates modern portfolio theory and optimization techniques using convex optimization, along with comprehensive risk analytics.

### Key Features:
- **Portfolio Optimization Strategies**: Maximum Sharpe Ratio, Minimum Variance, and Maximum Return portfolios
- **Efficient Frontier**: Calculation and visualization of optimal portfolios along the risk-return spectrum
- **Risk Analytics**: Multiple Value at Risk (VaR) methodologies including Parametric, Historical, Monte Carlo, Modified (Cornish-Fisher), and Conditional VaR
- **Real-time Data Integration**: Automated data retrieval and analysis for various asset classes
- **Interactive Visualizations**: Plotly-based charts for portfolio statistics and efficient frontier

### Files:
- `portfolio_optimizer.py`
- `risk_analytics.py`

## Module 3: Black-Scholes Option Pricing

This module provides a comprehensive framework for pricing various option types using analytical and numerical methods.

### Key Features:
- **Analytical Pricing**: Closed-form Black-Scholes pricing for European options
- **Monte Carlo Simulation**: Variance reduction techniques for pricing European, Asian, and Barrier options
- **Greeks Calculation**: Delta, Gamma, Vega, Theta, Rho for risk management
- **Implied Volatility**: Calculation from market prices
- **Market Data Integration**: Real-time options data retrieval
- **Visualization**: Interactive plots for option analysis

### Files:
- `bs_option_pricer.py`

## Module 4: Machine Learning

This module contains examples of optimization, regression, classification,
feature selection, decision trees, and boosting.

### Key Components:
- **Regression and Optimization**: Linear regression model comparisons, logistic regression, and batch gradient descent
- **Classification**: K-nearest neighbors, loan-status decision-tree classification, and market-trend logistic and XGBoost classifiers
- **Boosting**: A from-scratch residual-based regression boosting example and an XGBoost classification workflow with randomized hyperparameter search and time-series cross-validation
- **Feature Preparation and Selection**: Percentile-based outlier clipping, correlation filtering, VIF, SelectKBest, RFE, and RFECV
- **Model Evaluation**: Classification and regression metrics, visualizations, and a basic trading-signal analysis

See [`module_4/README.md`](./module_4/README.md) for an inventory of the Python modules and a detailed explanation of estimation, prediction, boosting formulas, parameter search, and the distinction between logloss and ROC-AUC.

## Requirements

- Python 3.x
- Required packages: pandas, numpy, matplotlib, scipy, cvxpy, plotly, yfinance, quantmod (and others as specified in individual modules)

## Usage

Each module contains standalone scripts. Refer to the individual README.md files in each module folder for detailed usage instructions and examples.