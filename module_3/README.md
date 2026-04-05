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
  - `moment_matcing`
  - `antithetic`
  - `sobol`
- Monte Carlo pricing for:
  - European options
  - Asian options
  - Barrier options
  - Binary options
  - Lookback options
- Greeks calculation for European options: Delta, Gamma, Vega, Theta, Rho
- Implied volatility estimation via binary search
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

### Implied Volatility

```python
from bs_option_pricer import compute_IV_from_price

implied_vol = compute_IV_from_price(
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

## Notes

- `calculate_greeks_for_market_data` is included in `bs_option_pricer.py` but its market-data workflow is partially implemented and may need additional completion.
- `option_pricer_fdm.py` uses an explicit finite-difference scheme, so choose `NAS`, `T`, and `dt` carefully for numerical stability.
- `quantmod` integration may require valid market access credentials or data availability.

## Getting Started

1. Install the required packages.
2. Run `bs_option_pricer.py` or `option_pricer_fdm.py` directly to test the example workflows.
3. Use the included functions and classes to compare analytic, Monte Carlo, and finite-difference option pricing.
