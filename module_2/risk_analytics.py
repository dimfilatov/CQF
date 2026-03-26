# -*- coding: utf-8 -*-
"""
Risk Analytics Module

Combined from var_analytics.py, var_backtesting.py, and var_calculator.py
Provides a comprehensive class for VaR calculations, backtesting, and volatility analysis.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import logging
from quantmod.markets import getData, getTicker
from scipy.stats import norm, stats, skew, kurtosis
from scipy.optimize import fsolve
from quantmod.risk import RiskInputs, VaRMetrics, VaRAnalyzer
from numpy.linalg import multi_dot
from tabulate import tabulate

logging.basicConfig(level=logging.INFO)

class RiskAnalytics:
    """
    A comprehensive class for risk analytics including VaR calculations,
    backtesting, and volatility analysis for portfolios and single assets.
    """

    def __init__(self, returns=None, tickers=None, period="max", confidence_levels=[0.90, 0.95, 0.99], num_simulations=100000):
        """
        Initialize the RiskAnalytics class.

        Parameters:
        -----------
        returns : pd.DataFrame or pd.Series
            Historical returns data. If None, will load from tickers.
        tickers : list or str
            List of tickers or single ticker to load data for.
        period : str
            Period for data loading (e.g., "max", "5y").
        confidence_levels : list
            List of confidence levels for VaR calculations.
        num_simulations : int
            Number of simulations for Monte Carlo VaR.
        """
        self.confidence_levels = confidence_levels
        self.num_simulations = num_simulations
        self.returns = returns
        self.tickers = tickers
        self.period = period

        if self.returns is None and self.tickers is not None:
            self.load_data()

    def load_data(self):
        """
        Load historical data from Yahoo Finance and compute returns.
        """
        df = getData(tickers=self.tickers, period=self.period)
        if isinstance(df.columns, pd.MultiIndex):
            df = df['Close']
        df = df.dropna()
        self.returns = df.pct_change().dropna()
        logging.info(f"Loaded data for {len(self.returns)} periods")

    def ewma_vol(self, r, decay):
        """
        Calculate Exponentially Weighted Moving Average (EWMA) volatility.

        Parameters:
        -----------
        r : array-like
            Array of returns.
        decay : float
            Decay factor for EWMA.

        Returns:
        --------
        pd.Series: EWMA volatility values.
        """
        vol = np.zeros(len(r))
        vol[0] = np.std(r)
        for i in range(1, len(r)):
            vol[i] = np.sqrt(decay * vol[i-1]**2 + (1 - decay) * r[i-1]**2)
        return pd.Series(vol, index=r.index)

    def vol_calculator(self, method="rw", lookback_period=30, decay=0.94):
        """
        Calculate volatility using specified method.

        Parameters:
        -----------
        method : str
            Method to use ('rw' for rolling window, 'ewma' for EWMA).
        lookback_period : int
            Lookback period for rolling window.
        decay : float
            Decay factor for EWMA.

        Returns:
        --------
        pd.Series: Volatility values.
        """
        if method == "rw":
            return self.returns.rolling(lookback_period, min_periods=lookback_period).std()
        elif method == "ewma":
            return self.ewma_vol(self.returns, decay)

    def parametric_var(self, asset=None):
        """
        Calculate Parametric VaR.

        Parameters:
        -----------
        asset : str, optional
            Specific asset column. If None, assumes single series or portfolio.

        Returns:
        --------
        dict: VaR values for each confidence level.
        """
        if asset:
            ret = self.returns[asset]
        else:
            ret = self.returns if isinstance(self.returns, pd.Series) else self.returns.mean(axis=1)

        mean = ret.mean()
        stdev = ret.std()
        var_dict = {}
        for cl in self.confidence_levels:
            z = norm.ppf(1 - cl)
            var_dict[f"{int(cl*100)}%"] = -(mean + z * stdev)  # Negative for loss
        return var_dict

    def historical_var(self, asset=None):
        """
        Calculate Historical VaR.

        Parameters:
        -----------
        asset : str, optional
            Specific asset column.

        Returns:
        --------
        dict: VaR values for each confidence level.
        """
        if asset:
            ret = self.returns[asset]
        else:
            ret = self.returns if isinstance(self.returns, pd.Series) else self.returns.mean(axis=1)

        var_dict = {}
        for cl in self.confidence_levels:
            var_dict[f"{int(cl*100)}%"] = -np.percentile(ret, (1 - cl) * 100)
        return var_dict

    def monte_carlo_var(self, asset=None):
        """
        Calculate Monte Carlo VaR.

        Parameters:
        -----------
        asset : str, optional
            Specific asset column.

        Returns:
        --------
        dict: VaR values for each confidence level.
        """
        if asset:
            ret = self.returns[asset]
        else:
            ret = self.returns if isinstance(self.returns, pd.Series) else self.returns.mean(axis=1)

        np.random.seed(42)
        mean = ret.mean()
        stdev = ret.std()
        sim_returns = np.random.normal(mean, stdev, self.num_simulations)

        var_dict = {}
        for cl in self.confidence_levels:
            var_dict[f"{int(cl*100)}%"] = -np.percentile(sim_returns, (1 - cl) * 100)
        return var_dict

    def modified_var(self, asset=None):
        """
        Calculate Modified (Cornish-Fisher) VaR.

        Parameters:
        -----------
        asset : str, optional
            Specific asset column.

        Returns:
        --------
        dict: Modified VaR values for each confidence level.
        """
        if asset:
            ret = self.returns[asset]
        else:
            ret = self.returns if isinstance(self.returns, pd.Series) else self.returns.mean(axis=1)

        mean = np.mean(ret)
        stdev = np.std(ret)
        s = skew(ret)
        k = kurtosis(ret)

        var_dict = {}
        for cl in self.confidence_levels:
            z = abs(norm.ppf(1 - cl))
            t = z + (1/6)*(z**2 - 1)*s + (1/24)*(z**3 - 3*z)*k - (1/36)*(2*z**3 - 5*z)*s**2
            var_dict[f"{int(cl*100)}%"] = -(mean - t * stdev)
        return var_dict

    def conditional_var(self, asset=None):
        """
        Calculate Conditional VaR (Expected Shortfall).

        Parameters:
        -----------
        asset : str, optional
            Specific asset column.

        Returns:
        --------
        dict: CVaR values for each confidence level.
        """
        if asset:
            ret = self.returns[asset]
        else:
            ret = self.returns if isinstance(self.returns, pd.Series) else self.returns.mean(axis=1)

        var_dict = {}
        for cl in self.confidence_levels:
            var_threshold = -np.percentile(ret, (1 - cl) * 100)
            cvar = ret[ret <= -var_threshold].mean()
            var_dict[f"{int(cl*100)}%"] = -cvar
        return var_dict

    def portfolio_var(self, weights):
        """
        Calculate Parametric VaR for a portfolio.

        Parameters:
        -----------
        weights : array-like
            Portfolio weights.

        Returns:
        --------
        dict: Portfolio VaR values.
        """
        port_returns = self.returns.dot(weights)
        mean = port_returns.mean()
        stdev = port_returns.std()
        var_dict = {}
        for cl in self.confidence_levels:
            z = norm.ppf(1 - cl)
            var_dict[f"{int(cl*100)}%"] = -(mean + z * stdev)
        return var_dict

    def backtest_var(self, var_series, actual_returns, confidence_level=0.99):
        """
        Backtest VaR using the quantmod VaRAnalyzer.

        Parameters:
        -----------
        var_series : pd.Series
            Series of VaR values.
        actual_returns : pd.Series
            Series of actual returns.
        confidence_level : float
            Confidence level for backtesting.

        Returns:
        --------
        dict: Backtesting results.
        """
        analyzer = VaRAnalyzer(
            inputs=RiskInputs(
                portfolio_returns=actual_returns.to_frame(),
                confidence_level=confidence_level,
                lookback_period=len(actual_returns),
            )
        )
        results = analyzer.run
        return results

    def compute_rolling_var(self, asset=None, window=30):
        """
        Compute rolling parametric VaR.

        Parameters:
        -----------
        asset : str, optional
            Specific asset column.
        window : int
            Rolling window size.

        Returns:
        --------
        pd.Series: Rolling VaR values.
        """
        if asset:
            ret = self.returns[asset]
        else:
            ret = self.returns if isinstance(self.returns, pd.Series) else self.returns.mean(axis=1)

        rolling_mean = ret.rolling(window).mean()
        rolling_std = ret.rolling(window).std()
        z = norm.ppf(0.01)  # For 99% VaR
        rolling_var = -(rolling_mean + z * rolling_std)
        return rolling_var

    def print_var_table(self, var_dict, title="VaR Results"):
        """
        Print VaR results in a tabular format.

        Parameters:
        -----------
        var_dict : dict
            Dictionary of VaR values.
        title : str
            Table title.
        """
        table = [[k, f"{v:.4f}"] for k, v in var_dict.items()]
        print(f"\n{title}")
        print(tabulate(table, headers=["Confidence Level", "VaR"], tablefmt="grid"))

# Example usage
if __name__ == "__main__":
    # Example with single asset
    ra = RiskAnalytics(tickers="NVDA", period="2y")
    print("Parametric VaR:", ra.parametric_var())
    print("Historical VaR:", ra.historical_var())
    print("Monte Carlo VaR:", ra.monte_carlo_var())
    print("Modified VaR:", ra.modified_var())
    print("Conditional VaR:", ra.conditional_var())