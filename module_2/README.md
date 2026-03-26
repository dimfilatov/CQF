# Portfolio Optimization and Risk Analytics

A comprehensive Python framework for portfolio optimization using Modern Portfolio Theory and advanced risk analytics, including Value at Risk (VaR) calculations and backtesting.

## 📋 Overview

This project provides a complete solution for investment portfolio management, combining:

1. **Portfolio Optimization** using CVXPY convex optimization:
   - Maximum Sharpe Ratio Portfolio
   - Minimum Variance Portfolio
   - Maximum Return Portfolio
   - Efficient Frontier calculation

2. **Risk Analytics** with multiple VaR methodologies:
   - Parametric VaR
   - Historical VaR
   - Monte Carlo VaR
   - Modified (Cornish-Fisher) VaR
   - Conditional VaR (Expected Shortfall)
   - Portfolio VaR
   - VaR backtesting

The framework uses object-oriented design with `PortfolioOptimizer` and `RiskAnalytics` classes for modular, reusable code.

## 🎯 Key Features

- ✅ **Data Retrieval**: Yahoo Finance integration with yfinance for historical price data
- ✅ **Flexible Date Ranges**: Specify custom start/end dates for analysis periods
- ✅ **Multiple Asset Classes**: Support for equities, ETFs, commodities, forex, and crypto
- ✅ **Convex Optimization**: CVXPY-based portfolio optimization with multiple solvers
- ✅ **Risk Management**: Comprehensive VaR calculations with statistical backtesting
- ✅ **Interactive Visualizations**: Plotly charts for efficient frontier and portfolio compositions
- ✅ **Modular Architecture**: Separate classes for optimization and risk analysis
- ✅ **Statistical Testing**: Normality tests and distribution analysis

## 📚 Theoretical Background

### Modern Portfolio Theory (MPT)

Modern Portfolio Theory, introduced by Harry Markowitz, optimizes portfolios based on expected returns and risk.

**Key Concepts:**

#### Expected Portfolio Return
$$\mu_p = w^T \cdot \mu$$

#### Portfolio Variance
$$\sigma_p^2 = w^T \cdot \Sigma \cdot w$$

#### Sharpe Ratio
$$SR = \frac{\mu_p - r_f}{\sigma_p}$$

### Value at Risk (VaR)

VaR measures the maximum potential loss over a specific time horizon at a given confidence level.

**VaR Methodologies:**
- **Parametric**: Assumes normal distribution
- **Historical**: Uses empirical distribution
- **Monte Carlo**: Simulation-based
- **Modified**: Adjusts for skewness and kurtosis

## 🚀 Installation

```bash
pip install numpy pandas cvxpy yfinance plotly scipy quantmod tabulate
```

## 💡 Usage

### Basic Portfolio Optimization

```python
from portfolio_optimizer import PortfolioOptimizer

# Define instruments
instruments = [
    {"symbol": "AAPL", "name": "Apple Inc", ...},
    # Add more instruments
]

# Create optimizer
optimizer = PortfolioOptimizer(instruments, return_target=0.1)

# Load data
optimizer.load_data()

# Compute statistics
optimizer.compute_statistics()

# Optimize portfolios
msr_weights = optimizer.optimize_max_sharpe()
mv_weights = optimizer.optimize_min_variance()
mr_weights = optimizer.optimize_max_return()

# Visualize results
ef_port = optimizer.calculate_efficient_frontier()
optimizer.visualize(ef_port, msr_weights, mv_weights, mr_weights)
```

### Risk Analytics

```python
from risk_analytics import RiskAnalytics

# Initialize with returns data
ra = RiskAnalytics(returns=optimizer.returns)

# Calculate VaR for optimized portfolios
msr_var = ra.portfolio_var(msr_weights)
ra.print_var_table(msr_var, "MSR Portfolio VaR")
```

### Combined Workflow

```python
from portfolio_optimizer import main
main()  # Runs complete optimization + risk analysis
```

## 📊 Output

The framework generates:
- Optimized portfolio weights for different strategies
- Efficient frontier visualization
- VaR estimates at 90%, 95%, and 99% confidence levels
- Statistical backtesting results
- Interactive Plotly charts

## 🔧 Configuration

- **Date Range**: Modify `start_date` and `end_date` in `PortfolioOptimizer`
- **Confidence Levels**: Adjust in `RiskAnalytics` initialization
- **Risk-Free Rate**: Set in `PortfolioOptimizer` for Sharpe ratio
- **VaR Methods**: Choose from parametric, historical, Monte Carlo, modified

## 📈 Supported Assets

- Equities (stocks)
- ETFs
- Commodities
- Forex pairs
- Cryptocurrencies

## 🤝 Contributing

This project combines portfolio theory with practical risk management tools. The modular design allows for easy extension with additional optimization constraints or risk metrics.

## 📚 References

- Markowitz, H. (1952). Portfolio Selection
- Jorion, P. (2007). Value at Risk: The New Benchmark for Managing Financial Risk
- CVXPY Documentation: https://www.cvxpy.org/
- yfinance Documentation: https://pypi.org/project/yfinance/

---

**Note**: This framework is for educational and research purposes. Always consult with financial professionals before making investment decisions.

### Optimization Problems

**1. Maximum Sharpe Ratio**
- Maximizes risk-adjusted returns
- Solves a convex optimization problem by setting excess return = 1
- Normalized portfolio weights sum to 1

**2. Minimum Variance**
- Minimizes portfolio volatility
- Most conservative approach
- Useful for risk-averse investors

**3. Maximum Return**
- Maximizes expected returns
- May concentrate weights in highest-return assets
- Ignores risk considerations

## 🚀 Getting Started

### Requirements

- Google Colab account (free)
- Supabase credentials (SUPABASE_URL and SUPABASE_KEY)
- Internet connection

### Installation in Google Colab

1. **Open Google Colab**
   - Go to [https://colab.research.google.com](https://colab.research.google.com)

2. **Upload or Clone the Script**
   - Create a new notebook or upload `portfolio_theory_colab.py`
   - Or paste the code directly into a cell

3. **Set Up Credentials**
   - Click on the 🔑 Secrets icon in the left sidebar
   - Add two secrets:
     - Key: `SUPABASE_URL` → Value: Your Supabase URL
     - Key: `SUPABASE_KEY` → Value: Your Supabase API key
   - Click "Grant access" to allow the notebook to access these secrets

4. **Run the Script**
   - Execute the cells in order
   - The script will automatically install required packages
   - Results and visualizations will display directly in Colab


## 📊 Script Structure

### Section 0: Data Retrieval
- Installs required packages
- Authenticates with Supabase
- Retrieves historical price data for 8 assets
- Cleans and prepares data

### Section 1: Portfolio Statistics
- Calculates annualized returns and volatility
- Computes mean returns vector
- Constructs covariance matrix
- Displays summary statistics

### Section 2: Maximum Sharpe Ratio Portfolio
- Solves convex optimization problem
- Objective: Minimize portfolio variance
- Constraint: Excess return = 1 unit
- Normalizes weights to sum to 1
- Returns risk-adjusted optimal portfolio

### Section 3: Minimum Variance Portfolio
- Solves quadratic programming problem
- Objective: Minimize portfolio variance
- Constraints: Fully invested, no short-selling
- Most conservative portfolio allocation

### Section 4: Maximum Return Portfolio
- Solves linear programming problem
- Objective: Maximize expected return
- Constraints: Fully invested, no short-selling
- Often concentrates in highest-return assets

### Section 5: Efficient Frontier
- Calculates 50-100 optimal portfolios
- Each point minimizes risk for a target return
- Plots the risk-return tradeoff curve
- Shows all three optimized portfolios on the frontier

### Section 6-7: Visualizations
- Efficient frontier line plot
- Portfolio composition pie charts
- Interactive Plotly charts
- Performance metrics display

## 🏦 Assets Included

Default portfolio includes 8 assets across multiple classes:

| Asset | Symbol | Type | Exchange |
|-------|--------|------|----------|
| Apple Inc | AAPL | Stock | NASDAQ |
| NVIDIA Corp | NVDA | Stock | NASDAQ |
| JPMorgan Chase | JPM | Stock | NYSE |
| SPY (S&P 500 ETF) | SPY | ETF | NYSEARCA |
| SPDR Gold ETF | GLD | ETF | NYSEARCA |
| Treasury Bond ETF | TLT | ETF | NASDAQ |
| EUR/USD | EURUSD=X | Forex | OTC |
| Bitcoin | BTC-USD | Crypto | Coinbase |

### Visualizations
1. **Efficient Frontier Chart** - Shows the optimal risk-return curve
2. **Portfolio Composition Charts** - Pie charts for each portfolio strategy

## ⚙️ Customization

### Change Asset List
Edit the `instrument_list` in Section 0 to include different assets:

```python
instrument_list = [
    {
        "symbol": "YOUR_SYMBOL",
        "name": "Asset Name",
        "asset_class": "EQUITY",
        "instrument_type": "STOCK",
        "exchange": "NASDAQ"
    },
    # ... more assets
]
```

### Adjust Time Period
Modify the date range in the `load_history()` call:

```python
results = qm.load_history(instruments, "2020-01-01", "2026-03-05")
```

### Change Risk-Free Rate
Adjust the Sharpe ratio calculation:

```python
msr_weights = optimize_max_sharpe(mean_returns, cov_matrix, risk_free_rate=0.04)
```

### Increase Efficient Frontier Points
For smoother visualization:

```python
ef_curve = efficient_frontier(mean_returns, cov_matrix, points=200)
```

## 📦 Dependencies

| Package | Purpose |
|---------|---------|
| **cvxpy** | Convex optimization solver |
| **numpy** | Numerical computations |
| **pandas** | Data manipulation |
| **quantmod** | Financial data retrieval |
| **plotly** | Interactive visualizations |

## ⚠️ Important Notes

2. **Supabase Credentials**: Never commit or share your API keys. Use Colab's Secrets Manager.

3. **Data Quality**: The optimization is only as good as your data. Ensure historical prices are clean and complete.

4. **Assumptions**: The script assumes:
   - Normal distribution of returns
   - Annualized data (260 trading days/year)
   - No transaction costs or taxes
   - No constraints on asset allocation

5. **Performance**: Calculating the efficient frontier with many points may take time. Start with 50 points and increase as needed.

## 🔗 Resources

- **CVXPY Documentation**: [cvxpy.org](https://www.cvxpy.org/index.html)
- **Plotly Documentation**: [plotly.com/python](https://plotly.com/python/)
- **Quantmod Library**: [kannansingaravelu.com/quantmod](https://kannansingaravelu.com/quantmod/)
- **Modern Portfolio Theory**: [Investopedia MPT](https://www.investopedia.com/terms/m/modernportfoliotheory.asp)


## 🤝 Troubleshooting

### "Module not found" Error
- Ensure all packages are installed
- Restart the kernel in Colab
- Check internet connection

### "Credentials not found" Error
- Verify Supabase URL and Key are correctly saved in Secrets Manager
- Check that the notebook has access permissions

### "Infeasible Problem" Error
- Some target returns may be infeasible given constraints
- This is normal; the script skips infeasible points
- Try adjusting the date range or asset list

### Empty Visualizations
- Ensure data was successfully retrieved
- Check that asset prices contain no missing values
- Verify covariance matrix is positive definite
