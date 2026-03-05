# -*- coding: utf-8 -*-
"""5_portfolio_cvxpy_colab.py

Portfolio Optimisation using CVXPY for Google Colab

This script demonstrates portfolio optimization using Modern Portfolio Theory.
It calculates the Maximum Sharpe Ratio Portfolio, Minimum Variance Portfolio,
and Maximum Return Portfolio, then visualizes the Efficient Frontier.

To run in Google Colab:
1. Go to https://colab.research.google.com
2. Create a new notebook and upload this file or paste the code
3. Follow the authentication steps for database access
"""

# ============================================
# INSTALL PACKAGES AND IMPORT LIBRARIES
# ============================================

# Install required packages
import subprocess
import sys

def install(package):
    subprocess.check_call([sys.executable, "-m", "pip", "install", package])

install("cvxpy")
install("quantmod")
install("plotly")

print("Packages installed successfully!")

# ============================================
# IMPORT LIBRARIES
# ============================================

import numpy as np
import pandas as pd
import cvxpy as cp
from quantmod.db import QuantmodDB
import quantmod.charts
from quantmod.timeseries.performance import dailyReturn, volatility
import plotly.graph_objects as go

try:
    from google.colab import userdata
    COLAB_ENV = True
except ImportError:
    COLAB_ENV = False
    print("Warning: Not running in Google Colab. Secrets will need to be provided manually.")

# ============================================
# AUTHENTICATE WITH DATABASE
# ============================================

if COLAB_ENV:
    try:
        # Get secrets safely from Google Colab Secrets Manager
        SUPABASE_URL = userdata.get('SUPABASE_URL')
        SUPABASE_KEY = userdata.get('SUPABASE_KEY')
        print("Credentials loaded from Google Colab Secrets Manager")
    except Exception as e:
        print(f"Error retrieving credentials: {e}")
        raise
else:
    # For local testing, prompt for credentials
    SUPABASE_URL = input("Enter your SUPABASE_URL: ")
    SUPABASE_KEY = input("Enter your SUPABASE_KEY: ")

# ============================================
# SECTION 0: DATA RETRIEVAL
# ============================================

print("\n=== DATA RETRIEVAL ===\n")

# Initialize QuantmodDB
qm = QuantmodDB(
    supabase_url=SUPABASE_URL,
    supabase_key=SUPABASE_KEY
)

# Define instruments for portfolio
instrument_list = [
    {
        "symbol": "AAPL",
        "name": "APPLE INC",
        "asset_class": "EQUITY",
        "instrument_type": "STOCK",
        "exchange": "NASDAQ"
    },
    {
        "symbol": "NVDA",
        "name": "NVIDIA CORP",
        "asset_class": "EQUITY",
        "instrument_type": "STOCK",
        "exchange": "NASDAQ"
    },
    {
        "symbol": "JPM",
        "name": "JPMORGAN CHASE & CO",
        "asset_class": "EQUITY",
        "instrument_type": "STOCK",
        "exchange": "NYSE"
    },
    {
        "symbol": "SPY",
        "name": "SPDR S&P 500 ETF",
        "asset_class": "INDEX",
        "instrument_type": "ETF",
        "exchange": "NYSEARCA"
    },
    {
        "symbol": "GLD",
        "name": "SPDR Gold ETF",
        "asset_class": "COMMODITY",
        "instrument_type": "ETF",
        "exchange": "NYSEARCA"
    },
    {
        "symbol": "TLT",
        "name": "iShares 20+ Year Treasury Bond ETF",
        "asset_class": "Bond",
        "instrument_type": "ETF",
        "exchange": "NASDAQ"
    },
    {
        "symbol": "EURUSD=X",
        "name": "EURUSD",
        "asset_class": "CURRENCY",
        "instrument_type": "FOREX",
        "exchange": "OTC"
    },
    {
        "symbol": "BTC-USD",
        "name": "BTCUSD",
        "asset_class": "CRYPTO",
        "instrument_type": "CRYPTO",
        "exchange": "COINBASE"
    },
]

instruments = [d["symbol"] for d in instrument_list]
print("Instruments:", instruments)

# Register instruments and load historical data
qm.register(instrument_list)
results = qm.load_history(instruments, "2019-01-01", "2026-03-05")

# Retrieve asset prices
df = (
    qm.get_asset_prices()
    # .drop(columns=["EURUSD=X", "TLT"])
    .dropna()
)

print("\nAsset Prices (last 5 rows):")
print(df.tail())

# ============================================
# SECTION 1: COMPUTE PORTFOLIO STATISTICS
# ============================================

print("\n=== PORTFOLIO STATISTICS ===\n")

# Calculate returns and volatility
returns = df.pct_change().dropna()
annual_returns = round(returns.mean() * 260 * 100, 2)
annual_stdev = round(returns.std() * np.sqrt(260) * 100, 2)

# Create statistics dataframe
stats = pd.DataFrame({
    'AnnRet': annual_returns,
    'AnnVol': annual_stdev
})

print("Annualized Returns and Volatility:")
print(stats)

# Compute statistics for optimization
mean_returns = (returns.mean() * 260).values
cov_matrix = (returns.cov() * 260).values
n = len(mean_returns)

print(f"\nNumber of assets: {n}")
print(f"Mean returns shape: {mean_returns.shape}")
print(f"Covariance matrix shape: {cov_matrix.shape}")

# ============================================
# SECTION 2: MAXIMUM SHARPE RATIO PORTFOLIO
# ============================================

print("\n=== MAXIMUM SHARPE RATIO PORTFOLIO ===\n")

def optimize_max_sharpe(mean_returns, cov_matrix, risk_free_rate=0.0):
    """
    Optimize portfolio to maximize Sharpe ratio.
    
    The Sharpe ratio is the excess return per unit of risk.
    We transform this non-convex problem into a convex one by setting
    excess return = 1 and minimizing variance.
    
    Parameters:
    -----------
    mean_returns : array
        Expected returns of assets
    cov_matrix : array
        Covariance matrix of asset returns
    risk_free_rate : float
        Risk-free rate (default: 0.0)
    
    Returns:
    --------
    w_normalized : array
        Optimal portfolio weights
    """
    w = cp.Variable(n)
    excess_return = mean_returns - risk_free_rate
    port_return = excess_return @ w
    port_risk = cp.quad_form(w, cov_matrix)
    
    # Minimize risk for 1 unit of excess return
    objective = cp.Minimize(port_risk)
    constraints = [w >= 0, port_return == 1]
    
    prob = cp.Problem(objective, constraints)
    prob.solve()
    
    if w.value is not None:
        # Normalize weights to sum to 1
        w_normalized = w.value / np.sum(w.value)
        return w_normalized
    else:
        return None

# Check available solvers
from cvxpy import installed_solvers
print("Available solvers:", installed_solvers())

# Run optimization
msr_weights = optimize_max_sharpe(mean_returns, cov_matrix)
print("\nMaximum Sharpe Ratio Portfolio Weights:")
msr_df = pd.DataFrame(msr_weights, index=df.columns, columns=['Weight'])
print(msr_df)

# ============================================
# SECTION 3: MINIMUM VARIANCE PORTFOLIO
# ============================================

print("\n=== MINIMUM VARIANCE PORTFOLIO ===\n")

def optimize_min_variance(cov_matrix):
    """
    Optimize portfolio to minimize variance.
    
    The minimum variance portfolio finds the asset allocation that
    reduces portfolio risk to its lowest possible level.
    
    Parameters:
    -----------
    cov_matrix : array
        Covariance matrix of asset returns
    
    Returns:
    --------
    w : array
        Optimal portfolio weights
    """
    w = cp.Variable(n)
    
    objective = cp.Minimize(cp.quad_form(w, cov_matrix))
    constraints = [cp.sum(w) == 1, w >= 0]
    
    prob = cp.Problem(objective, constraints)
    prob.solve()
    
    return w.value

# Run optimization
mv_weights = optimize_min_variance(cov_matrix)
print("Minimum Variance Portfolio Weights:")
mv_df = pd.DataFrame(mv_weights, index=df.columns, columns=['Weight'])
print(mv_df)

# ============================================
# SECTION 4: MAXIMUM RETURN PORTFOLIO
# ============================================

print("\n=== MAXIMUM RETURN PORTFOLIO ===\n")

def optimize_max_return(mean_returns):
    """
    Optimize portfolio to maximize expected return.
    
    This portfolio allocates all weight to the asset with the
    highest expected return.
    
    Parameters:
    -----------
    mean_returns : array
        Expected returns of assets
    
    Returns:
    --------
    w : array
        Optimal portfolio weights
    """
    w = cp.Variable(n)
    
    objective = cp.Maximize(mean_returns @ w)
    constraints = [cp.sum(w) == 1, w >= 0]
    
    prob = cp.Problem(objective, constraints)
    prob.solve()
    
    return w.value

# Run optimization
mr_weights = optimize_max_return(mean_returns)
print("Maximum Return Portfolio Weights:")
mr_df = pd.DataFrame(mr_weights, index=df.columns, columns=['Weight'])
print(mr_df)

# ============================================
# SECTION 5: EFFICIENT FRONTIER
# ============================================

print("\n=== EFFICIENT FRONTIER ===\n")

def efficient_frontier(mean_returns, cov_matrix, points=100):
    """
    Calculate the efficient frontier.
    
    The efficient frontier is the set of optimal portfolios that offer
    the highest expected return for a given level of risk.
    
    Parameters:
    -----------
    mean_returns : array
        Expected returns of assets
    cov_matrix : array
        Covariance matrix of asset returns
    points : int
        Number of points to calculate along the frontier
    
    Returns:
    --------
    frontier : array
        Array of (volatility, return) tuples along the frontier
    """
    target_returns = np.linspace(mean_returns.min(), mean_returns.max(), points)
    frontier = []
    
    for target in target_returns:
        w = cp.Variable(n)
        port_risk = cp.quad_form(w, cov_matrix)
        
        objective = cp.Minimize(port_risk)
        constraints = [cp.sum(w) == 1, w >= 0, mean_returns @ w == target]
        
        prob = cp.Problem(objective, constraints)
        prob.solve()
        
        if w.value is not None:
            vol = np.sqrt(w.value.T @ cov_matrix @ w.value)
            frontier.append((vol, target))
    
    return np.array(frontier)

# Calculate efficient frontier
print("Calculating efficient frontier...")
ef_curve = efficient_frontier(mean_returns, cov_matrix, points=50)
ef_port = 100 * pd.DataFrame(ef_curve, columns=['Volatility', 'Return'])
print(f"Efficient frontier calculated with {len(ef_port)} points")

# ============================================
# SECTION 6: PORTFOLIO STATISTICS FUNCTION
# ============================================

def get_stats(w):
    """
    Calculate portfolio return and volatility.
    
    Parameters:
    -----------
    w : array
        Portfolio weights
    
    Returns:
    --------
    ret : float
        Expected portfolio return (annualized, %)
    vol : float
        Portfolio volatility (annualized, %)
    """
    ret = mean_returns @ w
    vol = np.sqrt(w.T @ cov_matrix @ w)
    return 100 * ret, 100 * vol

# Calculate statistics for each optimized portfolio
msr_ret, msr_vol = get_stats(msr_weights)
mv_ret, mv_vol = get_stats(mv_weights)
mr_ret, mr_vol = get_stats(mr_weights)

print(f"\nMaximum Sharpe Ratio Portfolio:")
print(f"  Return: {msr_ret:.2f}%")
print(f"  Volatility: {msr_vol:.2f}%")
print(f"  Sharpe Ratio: {msr_ret/msr_vol:.2f}")

print(f"\nMinimum Variance Portfolio:")
print(f"  Return: {mv_ret:.2f}%")
print(f"  Volatility: {mv_vol:.2f}%")
print(f"  Sharpe Ratio: {mv_ret/mv_vol:.2f}")

print(f"\nMaximum Return Portfolio:")
print(f"  Return: {mr_ret:.2f}%")
print(f"  Volatility: {mr_vol:.2f}%")
print(f"  Sharpe Ratio: {mr_ret/mr_vol:.2f}")

# ============================================
# SECTION 7: VISUALIZATIONS
# ============================================

print("\n=== GENERATING VISUALIZATIONS ===\n")

# Create Efficient Frontier plot with Plotly
fig = go.Figure()

# Add efficient frontier
fig.add_trace(go.Scatter(
    x=ef_port['Volatility'],
    y=ef_port['Return'],
    mode='lines',
    name='Efficient Frontier',
    line=dict(color='black', width=2)
))

# Add Maximum Sharpe Ratio portfolio
fig.add_trace(go.Scatter(
    x=[msr_vol],
    y=[msr_ret],
    mode='markers+text',
    name='Max Sharpe Ratio',
    marker=dict(size=15, color='green', symbol='star'),
    text=['Max Sharpe'],
    textposition='top center'
))

# Add Minimum Variance portfolio
fig.add_trace(go.Scatter(
    x=[mv_vol],
    y=[mv_ret],
    mode='markers+text',
    name='Min Variance',
    marker=dict(size=15, color='blue', symbol='diamond'),
    text=['Min Variance'],
    textposition='top center'
))

# Add Maximum Return portfolio
fig.add_trace(go.Scatter(
    x=[mr_vol],
    y=[mr_ret],
    mode='markers+text',
    name='Max Return',
    marker=dict(size=15, color='red', symbol='square'),
    text=['Max Return'],
    textposition='top center'
))

# Update layout
fig.update_layout(
    title='Efficient Frontier Portfolio Optimization',
    xaxis_title='Volatility (Risk) %',
    yaxis_title='Expected Return %',
    hovermode='closest',
    width=900,
    height=600
)

fig.show()

# Portfolio composition pie charts
fig_msr = go.Figure(data=[go.Pie(
    labels=df.columns,
    values=msr_weights,
    title='Maximum Sharpe Ratio Portfolio'
)])
fig_msr.show()

fig_mv = go.Figure(data=[go.Pie(
    labels=df.columns,
    values=mv_weights,
    title='Minimum Variance Portfolio'
)])
fig_mv.show()

print("\n=== OPTIMIZATION COMPLETE ===\n")
print("All visualizations have been generated.")