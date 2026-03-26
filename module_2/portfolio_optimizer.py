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
# IMPORT LIBRARIES
# ============================================

import numpy as np
import pandas as pd
import cvxpy as cp
from quantmod.markets import getData
import plotly.graph_objects as go
from risk_analytics import RiskAnalytics


class PortfolioOptimizer:
    """
    A class for portfolio optimization using Modern Portfolio Theory.
    """

    def __init__(self, instruments, return_target, start_date="2019-01-01", end_date="2026-03-05", risk_free_rate=0.0):
        """
        Initialize the PortfolioOptimizer.

        Parameters:
        -----------
        instruments : list of dict
            List of instrument dictionaries with symbol, name, etc.
        start_date : str
            Start date for data download (YYYY-MM-DD)
        end_date : str
            End date for data download (YYYY-MM-DD)
        risk_free_rate : float
            Risk-free rate for Sharpe ratio calculation
        """
        self.return_target = return_target
        self.instruments = instruments
        self.symbols = [d["symbol"] for d in instruments]
        self.start_date = start_date
        self.end_date = end_date
        self.risk_free_rate = risk_free_rate
        self.df = None
        self.returns = None
        self.mean_returns = None
        self.cov_matrix = None
        self.n = None

    def load_data(self, period):
        """
        Load historical data from Yahoo Finance.
        """
        print("Instruments:", self.symbols)
        self.df = getData(tickers=self.symbols, period=period)
        # Extract closing prices
        if isinstance(self.df.columns, pd.MultiIndex):
            self.df = self.df['Close']
        self.df = self.df.dropna()
        print("\nAsset Prices (last 5 rows):")
        print(self.df.tail())

    def compute_statistics(self):
        """
        Compute portfolio statistics: returns, volatility, mean returns, covariance matrix.
        """
        print("\n=== PORTFOLIO STATISTICS ===\n")
        self.returns = self.df.pct_change().dropna()
        annual_returns = round(self.returns.mean() * 260 * 100, 2)
        annual_stdev = round(self.returns.std() * np.sqrt(260) * 100, 2)

        stats = pd.DataFrame({
            'AnnRet': annual_returns,
            'AnnVol': annual_stdev
        })
        print("Annualized Returns and Volatility:")
        print(stats)

        self.mean_returns = (self.returns.mean() * 260).values
        self.cov_matrix = (self.returns.cov() * 260).values
        self.n = len(self.mean_returns)

        print(f"\nNumber of assets: {self.n}")
        print(f"Mean returns shape: {self.mean_returns.shape}")
        print(f"Covariance matrix shape: {self.cov_matrix.shape}")

    def optimize_max_sharpe(self):
        """
        Optimize portfolio to maximize Sharpe ratio.

        Returns:
        --------
        weights : array
            Optimal portfolio weights
        """
        print("\n=== MAXIMUM SHARPE RATIO PORTFOLIO ===\n")
        w = cp.Variable(self.n)
        excess_return = self.mean_returns - self.risk_free_rate
        port_return = excess_return @ w
        port_risk = cp.quad_form(w, self.cov_matrix)

        objective = cp.Minimize(port_risk)
        constraints = [w >= 0, port_return == self.return_target]

        prob = cp.Problem(objective, constraints)
        prob.solve()

        if w.value is not None:
            w_normalized = w.value / np.sum(w.value)
            print("Maximum Sharpe Ratio Portfolio Weights:")
            msr_df = pd.DataFrame(w_normalized, index=self.df.columns, columns=['Weight'])
            print(msr_df)
            return w_normalized
        else:
            return None

    def optimize_min_variance(self):
        """
        Optimize portfolio to minimize variance.

        Returns:
        --------
        weights : array
            Optimal portfolio weights
        """
        print("\n=== MINIMUM VARIANCE PORTFOLIO ===\n")
        w = cp.Variable(self.n)

        objective = cp.Minimize(cp.quad_form(w, self.cov_matrix))
        constraints = [cp.sum(w) == 1, w >= 0]

        prob = cp.Problem(objective, constraints)
        prob.solve()

        print("Minimum Variance Portfolio Weights:")
        mv_df = pd.DataFrame(w.value, index=self.df.columns, columns=['Weight'])
        print(mv_df)
        return w.value

    def optimize_max_return(self):
        """
        Optimize portfolio to maximize expected return.

        Returns:
        --------
        weights : array
            Optimal portfolio weights
        """
        print("\n=== MAXIMUM RETURN PORTFOLIO ===\n")
        w = cp.Variable(self.n)

        objective = cp.Maximize(self.mean_returns @ w)
        constraints = [cp.sum(w) == 1, w >= 0]

        prob = cp.Problem(objective, constraints)
        prob.solve()

        print("Maximum Return Portfolio Weights:")
        mr_df = pd.DataFrame(w.value, index=self.df.columns, columns=['Weight'])
        print(mr_df)
        return w.value

    def calculate_efficient_frontier(self, points=50):
        """
        Calculate the efficient frontier.

        Parameters:
        -----------
        points : int
            Number of points to calculate along the frontier

        Returns:
        --------
        frontier : DataFrame
            DataFrame with Volatility and Return columns
        """
        print("\n=== EFFICIENT FRONTIER ===\n")
        target_returns = np.linspace(self.mean_returns.min(), self.mean_returns.max(), points)
        frontier = []

        for target in target_returns:
            w = cp.Variable(self.n)
            port_risk = cp.quad_form(w, self.cov_matrix)

            objective = cp.Minimize(port_risk)
            constraints = [cp.sum(w) == 1, w >= 0, self.mean_returns @ w == target]

            prob = cp.Problem(objective, constraints)
            prob.solve()

            if w.value is not None:
                vol = np.sqrt(w.value.T @ self.cov_matrix @ w.value)
                frontier.append((vol, target))

        ef_curve = np.array(frontier)
        ef_port = 100 * pd.DataFrame(ef_curve, columns=['Volatility', 'Return'])
        print(f"Efficient frontier calculated with {len(ef_port)} points")
        return ef_port

    def get_portfolio_stats(self, weights):
        """
        Calculate portfolio return and volatility.

        Parameters:
        -----------
        weights : array
            Portfolio weights

        Returns:
        --------
        ret : float
            Expected portfolio return (annualized, %)
        vol : float
            Portfolio volatility (annualized, %)
        """
        ret = self.mean_returns @ weights
        vol = np.sqrt(weights.T @ self.cov_matrix @ weights)
        return 100 * ret, 100 * vol

    def visualize(self, ef_port, msr_weights, mv_weights, mr_weights):
        """
        Generate visualizations for the efficient frontier and portfolio compositions.
        """
        print("\n=== GENERATING VISUALIZATIONS ===\n")

        msr_ret, msr_vol = self.get_portfolio_stats(msr_weights)
        mv_ret, mv_vol = self.get_portfolio_stats(mv_weights)
        mr_ret, mr_vol = self.get_portfolio_stats(mr_weights)

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
            labels=self.df.columns,
            values=msr_weights,
            title='Maximum Sharpe Ratio Portfolio'
        )])
        fig_msr.show()

        fig_mv = go.Figure(data=[go.Pie(
            labels=self.df.columns,
            values=mv_weights,
            title='Minimum Variance Portfolio'
        )])
        fig_mv.show()

        print("\n=== OPTIMIZATION COMPLETE ===\n")
        print("All visualizations have been generated.")


def main():
    """
    Main function to run the portfolio optimization.
    """
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

    # Check available solvers
    from cvxpy import installed_solvers
    print("Available solvers:", installed_solvers())

    # Create optimizer instance
    optimizer = PortfolioOptimizer(instrument_list, return_target=0.1)

    # Load data
    optimizer.load_data(period="2y")

    # Compute statistics
    optimizer.compute_statistics()

    # Initialize Risk Analytics with the same returns data
    ra = RiskAnalytics(returns=optimizer.returns)

    # Optimize portfolios
    msr_weights = optimizer.optimize_max_sharpe()
    mv_weights = optimizer.optimize_min_variance()
    mr_weights = optimizer.optimize_max_return()

    # Calculate efficient frontier
    ef_port = optimizer.calculate_efficient_frontier()

    # Visualize results
    optimizer.visualize(ef_port, msr_weights, mv_weights, mr_weights)

    # Compute and display VaR for optimized portfolios
    print("\n=== VALUE AT RISK ANALYSIS ===\n")

    if msr_weights is not None:
        msr_var = ra.portfolio_var(msr_weights)
        print("Maximum Sharpe Ratio Portfolio VaR:")
        ra.print_var_table(msr_var, "MSR Portfolio VaR")

    mv_var = ra.portfolio_var(mv_weights)
    print("Minimum Variance Portfolio VaR:")
    ra.print_var_table(mv_var, "MV Portfolio VaR")

    mr_var = ra.portfolio_var(mr_weights)
    print("Maximum Return Portfolio VaR:")
    ra.print_var_table(mr_var, "MR Portfolio VaR")


if __name__ == "__main__":
    main()