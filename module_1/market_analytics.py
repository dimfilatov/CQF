"""
This module provides functions for loading, processing, and analyzing market data.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from quantmod.markets import getData, getTicker

class MarketAnalytics:
    """
    A class to encapsulate financial data analytics for financial time series.
    """
    def __init__(self, 
                ticker: str, 
                period: str):
        self.period = period
        self.ticker = ticker

    def load_data(self):
        """
        Load data from Yahoo Finance.

        Returns:
            pd.DataFrame: DataFrame containing SPX data
            int: Number of records in the dataset
        """
        df = getData(tickers = self.ticker, period=self.period)

        n = len(df)
        print(f"Loaded {n} records from {self.ticker}")
        return df, n


    def create_lagged_features(self,df):
        """
        Create lagged price features for different time periods.
            
        Returns:
            pd.DataFrame: DataFrame with new St_1, St_2, St_5 columns
        """
        df["St_1"] = df["Close"].shift(1)
        df["St_2"] = df["Close"].shift(2)
        df["St_5"] = df["Close"].shift(5)
        return df


    def calculate_returns(self, df):
        """
        Calculate simple and log returns for different periods.
            
        Returns:
            pd.DataFrame: DataFrame with return columns (rt_1, rt_2, rt_5, rt_1_ln)
        """
        df["rt_1"] = df["Close"] / df["St_1"] - 1
        df["rt_2"] = df["Close"] / df["St_2"] - 1
        df["rt_5"] = df["Close"] / df["St_5"] - 1
        df["rt_1_ln"] = np.log(df["Close"] / df["St_1"])
        return df

    def calculate_volatility_stats(self, df):
        """
        Calculate and display adjusted volatility for different time periods.
        
        Args:
            df (pd.DataFrame): DataFrame with return columns
            
        Returns:
            dict: Dictionary containing adjusted standard deviations by period
        """
        std = {}
        days = [1, 2, 5]
        
        for i in days:
            std[i] = df["rt_" + str(i)].std()
            if i > 1:
                std[i] = std[i] / np.sqrt(i)
            print(f"Adjusted {i}-days std: {std[i]}")
        
        return std


    def analyze_even_odd_returns(self, df):
        """
        Analyze returns on even and odd indexed days.
        
        Args:
            df (pd.DataFrame): DataFrame with 'rt_1' column
        """
        even = df["rt_1"].iloc[0::2]
        odd = df["rt_1"].iloc[1::2]
        
        print(f"Even days std: {even.std()}")
        print(f"Odd days std: {odd.std()}")
        
        return even, odd


    def create_normal_sample(self, df):
        """
        Generate a normal distribution sample for comparison.
        
        Args:
            df (pd.DataFrame): DataFrame (for size reference)
            
        Returns:
            np.ndarray: Array of normally distributed random numbers
        """
        return np.random.randn(len(df))


    def plot_qq_analysis(self, df):
        """
        Create Q-Q plots for 1-day and 5-day returns.
        
        Args:
            df (pd.DataFrame): DataFrame with return columns
        """
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        
        # Q-Q plot for 1-day returns
        stats.probplot(df["rt_1"], dist="norm", plot=axes[0])
        axes[0].set_title("Q-Q Plot: 1-Day Returns")
        
        # Q-Q plot for 5-day returns
        stats.probplot(df["rt_5"], dist="norm", plot=axes[1])
        axes[1].set_title("Q-Q Plot: 5-Day Returns")
        
        plt.tight_layout()
        plt.show()


    def plot_return_distribution(self, df, normal):
        """
        Plot histogram comparing actual returns with normal distribution.
        
        Args:
            df (pd.DataFrame): DataFrame with returns
            normal (np.ndarray): Normal distribution sample
        """
        plt.figure(figsize=(10, 5))
        
        rt_1_scaled = (df["rt_1"] - df["rt_1"].mean()) / df["rt_1"].std()
        
        plt.hist(rt_1_scaled, range=(-7, 7), bins=100, alpha=0.7, 
                label="1-Day Returns", color="skyblue")
        plt.hist(normal, range=(-7, 7), bins=100, alpha=0.7, 
                label="Normal Distribution", color="salmon")
        
        plt.xlabel("Value")
        plt.ylabel("Frequency")
        plt.title("1-Day Returns vs Normal Distribution")
        plt.legend()
        plt.show()


    def plot_log_return_distribution(self, df, normal):
        """
        Plot histogram comparing log returns with normal distribution.
        
        Args:
            df (pd.DataFrame): DataFrame with log returns
            normal (np.ndarray): Normal distribution sample
        """
        df["rt_1_ln_scaled"] = (df["rt_1_ln"] - df["rt_1_ln"].mean()) / df["rt_1_ln"].std()
        
        plt.figure(figsize=(10, 5))
        
        plt.hist(df["rt_1_ln_scaled"], range=(-7, 7), bins=100, alpha=0.7, 
                label="Log Returns", color="skyblue")
        plt.hist(normal, range=(-7, 7), bins=100, alpha=0.7, 
                label="Normal Distribution", color="salmon")
        
        plt.xlabel("Value")
        plt.ylabel("Frequency")
        plt.title("Log Returns vs Normal Distribution")
        plt.legend()
        plt.show()
    
    def main(self):
        """
        Main method to execute the analysis workflow.
        """
        df, n = self.load_data()
        df = self.create_lagged_features(df)
        df = self.calculate_returns(df)
        std = self.calculate_volatility_stats(df)
        even, odd = self.analyze_even_odd_returns(df)
        normal_sample = self.create_normal_sample(df)
        self.plot_qq_analysis(df)
        self.plot_return_distribution(df, normal_sample)
        self.plot_log_return_distribution(df, normal_sample)

if __name__ == "__main__":
    # Example usage
    ticker = "^GSPC"  # S&P 500 index
    period = "max"     # Last 5 years of data
    analytics = MarketAnalytics(ticker, period)
    analytics.main()