# Portfolio Optimization using CVXPY

A comprehensive Google Colab-compatible Python script for optimizing investment portfolios using Modern Portfolio Theory and convex optimization techniques.

## 📋 Overview

This project demonstrates portfolio optimization using CVXPY, a Python library for convex optimization. The script calculates optimal portfolio weights for three different investment objectives:

1. **Maximum Sharpe Ratio Portfolio** - Maximizes risk-adjusted returns
2. **Minimum Variance Portfolio** - Minimizes overall portfolio risk
3. **Maximum Return Portfolio** - Maximizes expected returns

The script also computes and visualizes the **Efficient Frontier**, which represents the set of optimal portfolios along the risk-return spectrum.

## 🎯 Key Features

- ✅ Automated package installation for Google Colab
- ✅ Real-time data retrieval from Supabase
- ✅ Multiple portfolio optimization strategies
- ✅ Efficient frontier calculation
- ✅ Interactive Plotly visualizations
- ✅ Comprehensive portfolio statistics
- ✅ Support for equities, ETFs, commodities, forex, and crypto assets

## 📚 Theoretical Background

### Modern Portfolio Theory (MPT)

Modern Portfolio Theory, introduced by Harry Markowitz, is based on the principle that investors can construct portfolios that achieve optimal risk-return characteristics.

**Key Concepts:**

#### Expected Portfolio Return
$$\mu_p = w^T \cdot \mu$$

Where:
- $w$ = vector of portfolio weights
- $\mu$ = vector of expected asset returns

#### Portfolio Variance
$$\sigma_p^2 = w^T \cdot \Sigma \cdot w$$

Where:
- $\Sigma$ = covariance matrix of asset returns

#### Sharpe Ratio
$$SR = \frac{\mu_p - r_f}{\sigma_p}$$

Where:
- $r_f$ = risk-free rate

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
