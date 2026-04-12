# Black-Scholes Option Pricer

A Python project for option pricing using Black-Scholes analytics, Monte Carlo simulation, and finite-difference methods.

## Repository Modules

- `bs_option_pricer.py` — Black-Scholes pricing and Monte Carlo simulation for European, Asian, barrier, binary, and lookback options.
- `option_pricer_fdm.py` — Explicit finite-difference method for option pricing, with grid plotting and bilinear interpolation.

## Features

- Analytical Black-Scholes pricing for European call and put options
- Monte Carlo pricing methods:
  - `naive`
  - `exact`
  - `moment_matching`
  - `antithetic`
  - `sobol`
- Monte Carlo pricing for:
  - European options
  - Asian options
  - Barrier options
  - Binary options
  - Lookback options
- Greeks calculation for European options: Delta, Gamma, Vega, Theta, Rho
- Implied volatility (IV) estimation using two methods:
  - Binary search (`bisection_iv`)
  - Newton's method with analytical vega (`newton_iv`) — faster and more efficient
- Market data analysis:
  - Greeks calculation for real market options with implied volatility
  - Implied volatility term structure building across multiple maturities
  - IV skew visualization
- Finite-difference pricing using an explicit grid scheme
- Visualization helpers for path plots, option value grids, and 3D surfaces

## Installation

### Dependencies

```bash
pip install numpy pandas matplotlib scipy quantmod
```

## Usage

### Analytical and Monte Carlo Pricing

```python
from bs_option_pricer import BlackScholesOptionPricer

pricer = BlackScholesOptionPricer(
    S=100,
    K=100,
    T=1.0,
    t=0.0,
    r=0.05,
    sigma=0.2,
    dividend=0.02
)

call_price, put_price = pricer.analytical_price()
print(f"Call: {call_price:.4f}, Put: {put_price:.4f}")

call_mc, put_mc = pricer.monte_carlo_european_price(
    n_pths=20000,
    n_steps=100,
    method='exact'
)
print(f"Monte Carlo Call: {call_mc:.4f}, Put: {put_mc:.4f}")
```

### Greeks Calculation

```python
delta, gamma, vega, theta, rho = pricer.calculate_greeks_for_european('call')
print(delta, gamma, vega, theta, rho)
```

### Explicit Finite-Difference Pricing

```python
from option_pricer_fdm import option_price_with_finite_difference_method, plot_option_value

grid, s, t, intrinsic_values = option_price_with_finite_difference_method(
    K=100,
    T=1,
    r=0.05,
    sigma=0.2,
    call=True,
    NAS=100,
    american=False
)
plot_option_value(grid, s, t, intrinsic_values)
```

### Implied Volatility Estimation

#### Using Newton's Method (faster)
```python
from bs_option_pricer import newton_iv

implied_vol = newton_iv(
    option_price=10.0,
    spot_price=100,
    option_type='call',
    K=100,
    T=1.0,
    t=0.0,
    r=0.05,
    dividend=0.02
)
print(f"Implied volatility: {implied_vol:.4f}")
```

#### Using Binary Search
```python
from bs_option_pricer import bisection_iv

implied_vol = bisection_iv(
    option_price=10.0,
    spot_price=100,
    option_type='call',
    K=100,
    T=1.0,
    t=0.0,
    r=0.05,
    dividend=0.02
)
print(f"Implied volatility: {implied_vol:.4f}")
```

### Market Data Analysis

#### Calculate Greeks for Market Options
```python
from bs_option_pricer import calculate_greeks_for_market_data
from datetime import date

maturity = date(2026, 4, 30)
df = calculate_greeks_for_market_data(
    spot_price=658,
    ticker="SPY",
    maturity=maturity,
    strike_range=(650, 700),
    option_type='call',
    r=0.05,
    dividend=0.02,
    iv_method='newton'  # or 'bisection'
)
print(df)  # DataFrame with Implied Vol, Delta, Gamma, Vega, Theta, Rho
```

#### Build and Plot Implied Volatility Term Structure
```python
from bs_option_pricer import build_vol_term_structure, plot_vol_term_structure
from datetime import date

term_structure = build_vol_term_structure(
    spot_price=658,
    ticker="SPY",
    t_start=date(2026, 4, 12),
    t_end=date(2026, 5, 30),
    strike_range=(650, 700),
    option_type='call',
    r=0.05,
    dividend=0.02,
    iv_method='newton'
)
plot_vol_term_structure(term_structure)
```

## Implementation Details

### Implied Volatility Methods

- **Newton's Method** (`newton_iv`): Uses analytical vega from Black-Scholes Greeks for faster convergence. Includes:
  - Sigma bounds validation (0.001 to 5.0 = 0.1% to 500%)
  - Robust convergence criteria with both absolute and relative tolerance
  - Early termination if vega is near-zero
  - Better performance for typical market option prices

- **Binary Search** (`bisection_iv`): Robust but slower, guaranteed to converge

### Term Structure Analysis

- Filters available maturities based on date range and converts string dates to date objects
- Applies requested IV method (Newton or bisection) for consistency
- Returns strike-indexed Series for each maturity for easy visualization

## Notes

- `option_pricer_fdm.py` uses an explicit finite-difference scheme, so choose `NAS`, `T`, and `dt` carefully for numerical stability.
- `quantmod` integration may require valid market access credentials or data availability.
- Newton's method is recommended for IV calculation as it's ~10x faster than binary search when properly initialized.

## Getting Started

1. Install the required packages.
2. Run `bs_option_pricer.py` or `option_pricer_fdm.py` directly to test the example workflows.
3. Use the included functions and classes to compare analytic, Monte Carlo, and finite-difference option pricing.
