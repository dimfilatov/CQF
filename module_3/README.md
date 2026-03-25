# Black-Scholes Option Pricer

A comprehensive Python module for pricing European, Asian, and Barrier options using the Black-Scholes model, Monte Carlo simulation, and real market data integration.

## Features

- **Analytical Pricing**: Closed-form Black-Scholes pricing for European options
- **Monte Carlo Simulation**: Multiple variance reduction techniques (naive, exact, moment matching, antithetic, Sobol sequences)
- **Option Types**: European, Asian, and Barrier options
- **Greeks Calculation**: Delta, Gamma, Vega, Theta, Rho for risk management
- **Market Data Integration**: Real-time options data using quantmod
- **Implied Volatility**: Calculation from market prices using binary search
- **Visualization**: Interactive plots using plotly and matplotlib

## Installation

### Dependencies

```bash
pip install numpy pandas matplotlib scipy quantmod cufflinks plotly
```

### Required Libraries

- `numpy`: Numerical computations
- `pandas`: Data manipulation
- `matplotlib`: Static plotting
- `scipy`: Statistical functions and quasi-Monte Carlo
- `quantmod`: Financial market data
- `cufflinks`: Pandas integration with plotly
- `plotly`: Interactive plotting

## Quick Start

```python
from bs_option_pricer import BlackScholesOptionPricer

# Initialize pricer
pricer = BlackScholesOptionPricer(
    S=100,      # Spot price
    K=100,      # Strike price
    T=1.0,      # Time to maturity (years)
    t=0.0,      # Current time
    r=0.05,     # Risk-free rate
    sigma=0.2,  # Volatility
    dividend=0.02  # Dividend yield
)

# Get analytical prices
call_price, put_price = pricer.analytical_price()
print(f"Call: {call_price:.4f}, Put: {put_price:.4f}")

# Monte Carlo pricing
call_mc, put_mc = pricer.monte_carlo_european_price(
    n_pths=10000,  # Number of paths
    n_steps=50,    # Time steps
    method='exact' # Simulation method
)
```

## Class: BlackScholesOptionPricer

### Constructor Parameters

- `S` (float): Current stock price
- `K` (float): Strike price
- `T` (float): Time to maturity (in years)
- `t` (float): Current time (usually 0 for forward pricing)
- `r` (float): Risk-free interest rate (annualized)
- `sigma` (float): Volatility (annualized)
- `dividend` (float): Dividend yield (annualized)

### Methods

#### Analytical Pricing

```python
call_price, put_price = pricer.analytical_price()
```
Returns the closed-form Black-Scholes prices for call and put options.

#### Monte Carlo Pricing

##### European Options
```python
call_price, put_price = pricer.monte_carlo_european_price(n_pths, n_steps, method)
```

##### Asian Options
```python
call_price, put_price = pricer.monte_carlo_asian_price(n_pths, n_steps, method)
```

##### Barrier Options
```python
call_price, put_price = pricer.monte_carlo_barrier_price(n_pths, n_steps, method, barrier, barrier_type, rebate)
```

**Parameters:**
- `n_pths` (int): Number of Monte Carlo paths
- `n_steps` (int): Number of time steps per path
- `method` (str): Simulation method
  - `'naive'`: Euler discretization
  - `'exact'`: Exact GBM solution
  - `'moment_matching'`: Standardized normal variates
  - `'antithetic'`: Antithetic variates
  - `'sobol'`: Quasi-Monte Carlo with Sobol sequences
- `barrier` (float): Barrier level (for barrier options)
- `barrier_type` (str): `'up_and_out'`, `'down_and_out'`, `'up_and_in'`, `'down_and_in'`
- `rebate` (float): Rebate payment for knocked-out options

#### Greeks Calculation

```python
delta, gamma, vega, theta, rho = pricer.calculate_greeks_for_european(option_type)
```

**Parameters:**
- `option_type` (str): `'call'` or `'put'`

**Returns:** Tuple of (delta, gamma, vega, theta, rho)

#### Path Simulation

```python
S_T, paths = pricer.simulate_GBM_paths(n_pths, n_steps, method)
```

Returns final stock prices (`S_T`) and full path matrix (`paths`).

#### Visualization

```python
pricer.plot_paths(n_pths, n_steps, method)  # Plot sample paths
pricer.compare_methods_against_analytical(n_pths, n_steps)  # Compare methods
```

## Market Data Functions

### Calculate Greeks for Market Data

```python
calculate_greeks_for_market_data(spot_price, ticker, maturity, strike_range, option_type, r, dividend)
```

**Parameters:**
- `spot_price` (float): Current spot price
- `ticker` (str): Stock ticker symbol
- `maturity` (str): Option expiration date (YYYY-MM-DD)
- `strike_range` (tuple): (min_strike, max_strike)
- `option_type` (str): `'call'` or `'put'`
- `r` (float): Risk-free rate
- `dividend` (float): Dividend yield

This function:
1. Fetches real options data using quantmod
2. Calculates implied volatility for each strike
3. Computes Greeks using the implied volatility
4. Displays results in a table
5. Creates interactive plots of Greeks vs strikes

### Implied Volatility Calculation

```python
implied_vol = compute_IV_from_price(option_price, spot_price, option_type, K, T, t, r, dividend)
```

Calculates implied volatility using binary search algorithm.

## Examples

### Basic Option Pricing

```python
# Initialize with typical parameters
pricer = BlackScholesOptionPricer(S=150, K=155, T=0.5, t=0, r=0.03, sigma=0.25, dividend=0.01)

# Analytical pricing
call_analytical, put_analytical = pricer.analytical_price()

# Monte Carlo with different methods
methods = ['naive', 'exact', 'antithetic', 'sobol']
for method in methods:
    call_mc, put_mc = pricer.monte_carlo_european_price(50000, 100, method)
    print(f"{method}: Call={call_mc:.4f}, Put={put_mc:.4f}")
```

### Greeks Analysis

```python
# Calculate Greeks for different strikes
strikes = [90, 95, 100, 105, 110]
for strike in strikes:
    pricer = BlackScholesOptionPricer(S=100, K=strike, T=1, t=0, r=0.05, sigma=0.2, dividend=0.02)
    delta, gamma, vega, theta, rho = pricer.calculate_greeks_for_european('call')
    print(f"Strike {strike}: Delta={delta:.4f}, Gamma={gamma:.4f}")
```

### Real Market Data

```python
# Analyze SPY options
calculate_greeks_for_market_data(
    spot_price=450,
    ticker="SPY",
    maturity="2024-12-20",
    strike_range=(440, 460),
    option_type='call',
    r=0.042,
    dividend=0.012
)
```

### Barrier Options

```python
# Up-and-out call option
call_barrier, put_barrier = pricer.monte_carlo_barrier_price(
    n_pths=50000,
    n_steps=100,
    method='exact',
    barrier=120,  # Barrier level
    barrier_type='up_and_out',
    rebate=0.0    # No rebate
)
```

## Mathematical Background

### Black-Scholes Model

The Black-Scholes model assumes:
- Log-normal stock price dynamics: `dS = (r - q)S dt + σ S dW`
- Constant volatility, risk-free rate, and dividend yield
- No transaction costs or taxes
- Continuous trading

### Greeks Formulas

- **Delta**: ∂V/∂S
- **Gamma**: ∂²V/∂S²
- **Vega**: ∂V/∂σ
- **Theta**: ∂V/∂t
- **Rho**: ∂V/∂r

### Monte Carlo Methods

- **Naive**: Euler discretization of GBM
- **Exact**: Direct simulation using exact solution
- **Moment Matching**: Standardizes random variates to have exact mean 0, variance 1
- **Antithetic**: Uses pairs of positively and negatively correlated paths
- **Sobol**: Quasi-Monte Carlo using low-discrepancy sequences

## Performance Notes

- Monte Carlo methods with more paths provide better accuracy but take longer
- Variance reduction techniques (antithetic, moment matching, Sobol) improve convergence
- Analytical pricing is fastest and most accurate for European options
- Asian and barrier options require Monte Carlo simulation

## License

This module is provided for educational and research purposes.

## Contributing

Contributions are welcome! Please ensure code follows the existing style and includes appropriate documentation.