import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import norm
from scipy.stats import qmc, norm   
from quantmod.markets import getTicker
import quantmod.charts
from datetime import datetime, timedelta
import math


class BlackScholesOptionPricer:
    def __init__(self, S, K, T, t, r, sigma, dividend=0, T_f=None):
        self.S = S  # Current stock price
        self.K = K  # Strike price
        self.T = T  # Option maturity time
        self.t = t  # Current time
        self.T_f = T_f  # Time of futures delivery (optional)
        self.tau = self.T - self.t  # Time to option maturity
        self.tau_f = self.T_f - self.t if self.T_f is not None else None
        self.r = r  # Risk-free rate
        self.dividend = dividend  # Dividend yield
        self.sigma = sigma  # Volatility
        if self.tau <= 0:
            raise ValueError("Time to maturity must be positive")
        if self.sigma < 0:
            raise ValueError("Volatility must be non-negative")
        if self.sigma == 0:
            self.d1 = np.inf if self.S > self.K else -np.inf
            self.d2 = self.d1
        else:
            self.d1 = (np.log(self.S/self.K) + (self.r - self.dividend + 0.5 * self.sigma**2) * self.tau) / (self.sigma * np.sqrt(self.tau))
            self.d2 = self.d1 - self.sigma * np.sqrt(self.tau)

    def simulate_GBM_paths(self, n_pths, n_steps, method, asset_type=None):
        dt = self.tau / n_steps
        paths = {}
        paths = np.zeros((n_pths, n_steps+1))

        if asset_type == 'stock':
            paths[:, 0] = self.S
            if method == 'naive':
                for t in range(1, n_steps+1):
                    z = np.random.standard_normal(n_pths)
                    paths[:, t] = paths[:, t-1] * (1 + (self.r - self.dividend) * dt + self.sigma * np.sqrt(dt) * z)

            elif method == 'exact':
                for t in range(1, n_steps + 1):
                    z = np.random.standard_normal(n_pths)
                    paths[:, t] = paths[:, t-1] * np.exp((self.r - self.dividend - 0.5 * self.sigma **2) * dt + self.sigma * np.sqrt(dt) * z)
            elif method == 'moment_matching':
                z = np.random.standard_normal(n_pths)
                z_std = (z-np.mean(z)) / np.std(z)
                S_T = self.S * np.exp((self.r - self.dividend - 0.5 * self.sigma**2) * self.tau + self.sigma * np.sqrt(self.tau) * z_std)
            elif method == 'antithetic':
                z = np.random.standard_normal(n_pths//2)
                z_anti = -z 
                S_T = self.S * np.exp((self.r - self.dividend - 0.5 * self.sigma**2) * self.tau + self.sigma * np.sqrt(self.tau) * z)
                S_T_anti = self.S * np.exp((self.r - self.dividend - 0.5 * self.sigma**2) * self.tau + self.sigma * np.sqrt(self.tau) * z_anti)
                S_T = np.concatenate((S_T, S_T_anti))
            elif method == 'sobol':
                # Generate Sobol samples in [0,1], then map to standard normal
                n_sobol = 2 ** math.ceil(math.log2(n_pths))
                sobol = qmc.Sobol(d=1, scramble=True)
                u = sobol.random(n_sobol)
                z = norm.ppf(u.ravel()[:n_pths])  # Use first n_pths
                S_T = self.S * np.exp((self.r - self.dividend - 0.5 * self.sigma**2) * self.tau + self.sigma * np.sqrt(self.tau) * z)

            if method not in ['moment_matching', 'antithetic', 'sobol']:
                S_T = paths[:, -1]    
        
        return S_T, paths
   
    def analytical_price(self):
        call_price = np.exp(-self.dividend * (self.tau)) * self.S * norm.cdf(self.d1) - self.K * np.exp(- self.r * (self.tau)) * norm.cdf(self.d2)
        put_price = self.K * np.exp(-self.r * (self.tau)) * norm.cdf(-self.d2) - np.exp(-self.dividend * (self.tau)) * self.S * norm.cdf(-self.d1)
        return call_price, put_price
    
    def monte_carlo_european_price(self, n_pths, n_steps, method):
        S_T, paths = self.simulate_GBM_paths(n_pths, n_steps, method, asset_type='stock')
        call_payoffs = np.maximum(S_T - self.K, 0)
        put_payoffs = np.maximum(self.K - S_T, 0)
        call_price = np.exp(- self.r * self.tau) * np.mean(call_payoffs)
        put_price = np.exp(- self.r * self.tau) * np.mean(put_payoffs)
        return call_price, put_price

    def monte_carlo_asian_price(self, n_pths, n_steps, method):
        S_T, paths = self.simulate_GBM_paths(n_pths, n_steps, method, asset_type='stock')
        S_avg = np.mean(paths, axis = 1)
        payoffs_call = np.maximum(S_avg - self.K, 0)
        payoffs_put = np.maximum(self.K - S_avg, 0)
        call_price = np.exp(-self.r * self.tau) * np.mean(payoffs_call)
        put_price = np.exp(-self.r * self.tau) * np.mean(payoffs_put)
        return call_price, put_price
    
    def monte_carlo_barrier_price(self, n_pths, n_steps, method, barrier, barrier_type, rebate):
        S_T, paths = self.simulate_GBM_paths(n_pths, n_steps, method, asset_type='stock')
        if barrier_type == "up_and_out":
            valid_paths = ~np.any(paths > barrier, axis=1)
        elif barrier_type == "down_and_out":
            valid_paths = ~np.any(paths < barrier, axis=1)
        elif barrier_type == "up_and_in":
            valid_paths = np.any(paths > barrier, axis=1)
        elif barrier_type == "down_and_in":
            valid_paths = np.any(paths < barrier, axis = 1)
        else:            
            raise ValueError("Invalid barrier type")
        
        payoffs_call =np.where(valid_paths, np.maximum(S_T - self.K, 0), rebate)
        payoffs_put = np.where(valid_paths, np.maximum(self.K - S_T, 0), rebate)
        call_price = np.exp(-self.r * self.tau) * np.mean(payoffs_call)
        put_price = np.exp(-self.r * self.tau) * np.mean(payoffs_put)
        return call_price, put_price

    def monte_carlo_binary_price(self, n_pths, n_steps, method):
        S_T, paths = self.simulate_GBM_paths(n_pths, n_steps, method, asset_type='stock')
        payoffs_call =np.where(S_T - self.K > 0, 1, 0)
        payoffs_put = np.where(self.K-S_T > 0, 1,0)
        call_price = np.exp(-self.r * self.tau) * np.mean(payoffs_call)
        put_price = np.exp(-self.r * self.tau) * np.mean(payoffs_put)
        return call_price, put_price
    
    def monte_carlo_lookback_price(self, n_pths, n_steps, method):
        S_T, paths = self.simulate_GBM_paths(n_pths, n_steps, method, asset_type='stock')
        S_min = np.min(paths, axis=1)
        S_max = np.max(paths, axis=1)
        payoffs_call = np.maximum(S_T - S_min, 0)
        payoffs_put = np.maximum(S_max - S_T, 0)
        call_price = np.exp(-self.r * self.tau) * np.mean(payoffs_call)
        put_price = np.exp(-self.r * self.tau) * np.mean(payoffs_put)
        return call_price, put_price

    def plot_paths(self, n_pths, n_steps, method):
        S_T, paths = self.simulate_GBM_paths(n_pths, n_steps, method)
        plt.figure(figsize=(10,6))
        for i in range(min(n_pths, 100)):
            plt.plot(paths[i], lw=0.5)
        plt.title(f"GBM Paths - Method: {method}")
        plt.xlabel("Time Steps")
        plt.ylabel("Stock Price")
        plt.grid()
        plt.show()
    
    def compare_methods_against_analytical(self, n_pths, n_steps):
        methods = ['naive', 'exact', 'moment_matching', 'antithetic', 'sobol']
        results = []
        for method in methods:
            call_price, put_price = self.monte_carlo_european_price(n_pths, n_steps, method)
            results.append((method, call_price, put_price))
        analytical_call, analytical_put = self.analytical_price()
        results.append(('analytical', analytical_call, analytical_put))
        df = pd.DataFrame(results, columns=['Method', 'Call Price', 'Put Price'])
        print(df)

    def calculate_greeks_for_european(self, option_type):
        if self.tau == 0:
            gamma = 0
            theta = 0
            vega = 0
        else:
            gamma = np.exp(-self.dividend * self.tau) * norm.pdf(self.d1) / (self.sigma * self.S * np.sqrt(self.tau)) 
            vega = self.S * np.exp(-self.dividend * self.tau) * norm.pdf(self.d1) * np.sqrt(self.tau)
            if option_type == 'call':
                theta = (-self.S * norm.pdf(self.d1) * self.sigma / (2 * np.sqrt(self.tau)) - self.r * self.K * np.exp(-self.r * self.tau) * norm.cdf(self.d2)) / 365
            elif option_type == "put":
                theta = (-self.S * norm.pdf(self.d1) * self.sigma / (2 * np.sqrt(self.tau)) + self.r * self.K * np.exp(-self.r * self.tau) * norm.cdf(-self.d2)) / 365
            else:
                raise ValueError("Invalid option type")
        
        if option_type == 'call':
            delta = np.exp(-self.dividend * self.tau) * norm.cdf(self.d1)
            rho = self.K * self.tau * np.exp(-self.r * self.tau) * norm.cdf(self.d2) / 100
        elif option_type == "put":
            delta = np.exp(-self.dividend * self.tau) * (norm.cdf(self.d1) - 1)
            rho = -self.K * self.tau * np.exp(-self.r * self.tau) * norm.cdf(-self.d2) / 100
        else:
            raise ValueError("Invalid option type")
        
        return delta, gamma, vega, theta, rho

def bisection_iv(option_price, spot_price, option_type, K, T, t, r, dividend, tol=1e-6, max_iter=100):
    sigma_low = 1e-6
    sigma_high = 5.0
    for _ in range(max_iter):
        sigma_mid = (sigma_low + sigma_high) / 2
        pricer = BlackScholesOptionPricer(S=spot_price, K=K, T=T, t=t, r=r, sigma=sigma_mid, dividend=dividend)
        if option_type == 'call':
            price_mid, _ = pricer.analytical_price()
        else:
            _, price_mid = pricer.analytical_price()
        
        if abs(price_mid - option_price) < tol:
            return sigma_mid
        elif price_mid < option_price:
            sigma_low = sigma_mid
        else:
            sigma_high = sigma_mid
    return (sigma_low + sigma_high) / 2

def newton_iv(option_price, spot_price, option_type, K, T, t, r, dividend, tol=1e-6, max_iter=100):
    """Compute implied volatility using Newton's method with analytical vega.
    
    Args:
        option_price: Market price of the option
        spot_price: Current stock price
        option_type: 'call' or 'put'
        K: Strike price
        T: Maturity time
        t: Current time
        r: Risk-free rate
        dividend: Dividend yield
        tol: Convergence tolerance
        max_iter: Maximum iterations
    
    Returns:
        Implied volatility
    """
    if option_type not in ['call', 'put']:
        raise ValueError(f"Invalid option type: {option_type}. Must be 'call' or 'put'.")
    
    sigma = np.clip(0.2, 0.001, 5.0)  # Initial guess with bounds
    
    for iteration in range(max_iter):
        pricer = BlackScholesOptionPricer(S=spot_price, K=K, T=T, t=t, r=r, sigma=sigma, dividend=dividend)
        if option_type == 'call':
            price, _ = pricer.analytical_price()
        else:
            _, price = pricer.analytical_price()
        
        error = price - option_price
        
        # Use analytical vega from greeks instead of numerical derivative
        _, _, vega, _, _ = pricer.calculate_greeks_for_european(option_type)
        
        # Check convergence
        if abs(error) < tol:
            break
        
        # Avoid division by very small vega
        if abs(vega) < 1e-8:
            break
        
        sigma_new = sigma - error / vega
        sigma_new = np.clip(sigma_new, 0.001, 5.0)
        
        # Check convergence with both absolute and relative tolerance
        converged = abs(sigma_new - sigma) < max(tol, tol * abs(sigma))
        sigma = sigma_new
        
        if converged:
            break
    
    return sigma

def get_next_business_day():
    nextday = datetime.today()
    dte = 0
    while nextday.weekday() >= 5:
        nextday += timedelta(days=1)
    nextday = nextday.date()
    return nextday

def calculate_greeks_for_market_data(spot_price, ticker, maturity, strike_range, option_type, r, dividend, iv_method='newton'):
    spy = getTicker(ticker)
    dte = (maturity - datetime.today().date()).days / 365
    print(f"Calculating greeks for {ticker} options with maturity {maturity} and DTE {dte:.4f} years")
    options = spy.option_chain(maturity.strftime("%Y-%m-%d"))
    if option_type == 'call':
        df = options.calls[(options.calls['strike'] >= strike_range[0]) & (options.calls['strike'] <= strike_range[1]   )].copy()
    else:
        df = options.puts[(options.puts['strike'] >= strike_range[0]) & (options.puts['strike'] <= strike_range[1]   )].copy()
    df = df[['strike', 'lastPrice']]
    df.reset_index(drop=True, inplace=True)

    for idx, row in df.iterrows():
        if iv_method == 'newton':
            implied_vol = newton_iv(row['lastPrice'], spot_price, option_type, row['strike'], dte, 0, r, dividend)
        else:
            implied_vol = bisection_iv(row['lastPrice'], spot_price, option_type, row['strike'], dte, 0, r, dividend)
        opt = BlackScholesOptionPricer(S=spot_price, K=row['strike'], T=dte, t=0, r=r, sigma=implied_vol, dividend=dividend)
        greeks = opt.calculate_greeks_for_european(option_type)
        df.at[idx, 'Implied Vol'] = implied_vol
        df.at[idx, 'Delta'] = greeks[0]
        df.at[idx, 'Gamma'] = greeks[1]
        df.at[idx, 'Vega'] = greeks[2]
        df.at[idx, 'Theta'] = greeks[3]
        df.at[idx, 'Rho'] = greeks[4]
    # Set index
    df.set_index('strike', inplace=True)
    return df

def plot_greeks(df):
    print(df)    

    # now plot the greeks and implied vol against strikes
    plt.figure(figsize=(12, 8))
    plt.subplot(2, 3, 1)
    plt.plot(df.index, df['Delta'], marker='o')
    plt.title('Delta vs Strike')
    plt.xlabel('Strike Price')
    plt.ylabel('Delta')
    plt.grid()
    plt.subplot(2, 3, 2)
    plt.plot(df.index, df['Gamma'], marker='o')
    plt.title('Gamma vs Strike')
    plt.xlabel('Strike Price')
    plt.ylabel('Gamma')
    plt.grid()
    plt.subplot(2, 3, 3)
    plt.plot(df.index, df['Vega'], marker='o')
    plt.title('Vega vs Strike')
    plt.xlabel('Strike Price')

    plt.ylabel('Vega')
    plt.grid()
    plt.subplot(2, 3, 4)
    plt.plot(df.index, df['Theta'], marker='o')

    plt.title('Theta vs Strike')
    plt.xlabel('Strike Price')
    plt.ylabel('Theta')
    plt.grid()
    plt.subplot(2, 3, 5)
    plt.plot(df.index, df['Rho'], marker='o')
    plt.title('Rho vs Strike')
    plt.xlabel('Strike Price')
    plt.ylabel('Rho')
    plt.grid()
    plt.tight_layout()
    plt.show() 

def plot_iv_skew(df):

    plt.figure(figsize=(8, 5))
    plt.plot(df.index, df['Implied Vol'], marker='o')
    plt.title('Implied Volatility Skew')
    plt.xlabel('Strike Price')
    plt.ylabel('Implied Volatility')
    plt.grid()
    plt.show()

def build_vol_term_structure(spot_price, ticker, t_start, t_end, strike_range, option_type, r, dividend, iv_method='newton'):
    spy = getTicker(ticker)
    # Convert string dates from spy.options to date objects for comparison
    maturities = [datetime.strptime(m, '%Y-%m-%d').date() for m in spy.options 
                  if t_start <= datetime.strptime(m, '%Y-%m-%d').date() <= t_end]
    term_structure = {}
    for maturity in maturities:
        df = calculate_greeks_for_market_data(spot_price, ticker, maturity, strike_range, option_type, r, dividend, iv_method)
        term_structure[maturity] = df['Implied Vol']
    return term_structure

def plot_vol_term_structure(term_structure, include_strike=False):
    plt.figure(figsize=(10, 6))
    for maturity, iv_series in term_structure.items():
        plt.plot(iv_series.index, iv_series.values, marker='o', label=f'Maturity: {maturity}')
    plt.title('Implied Volatility Term Structure')
    plt.xlabel('Strike Price')
    plt.ylabel('Implied Volatility')
    plt.grid()
    plt.legend()
    plt.show()


if __name__ == "__main__":
    S = 100
    K = 100
    T = 1
    T_f = 2
    t = 0
    r = 0.05
    sigma = 0.2
    dividend = 0.02
    pricer = BlackScholesOptionPricer(S, K, T, t, r, sigma, dividend)
    pricer.compare_methods_against_analytical(n_pths=10000, n_steps=50)
    maturity_date = get_next_business_day()
    print(f"Next business day: {maturity_date}")
    df = calculate_greeks_for_market_data(spot_price=658, ticker="SPY", maturity=maturity_date, strike_range=(650, 700), option_type='call', r=0.05, dividend=0.02)
    plot_greeks(df)
    plot_iv_skew(df)
    term_structure = build_vol_term_structure(spot_price=658, ticker="SPY", t_start=datetime.today().date(), t_end=datetime.strptime('2026-04-30', '%Y-%m-%d').date(), strike_range=(650, 700), option_type='call', r=0.05, dividend=0.02)
    plot_vol_term_structure(term_structure)

    # calulate price of binary, barrier, asian and lookback options
    call_price, put_price = pricer.monte_carlo_binary_price(n_pths=10000, n_steps=50, method='exact')
    print(f"Binary Call Price: {call_price:.4f}, Binary Put Price: {put_price:.4f}")
    call_price, put_price = pricer.monte_carlo_barrier_price(n_pths=10000, n_steps=50, method='exact', barrier=110, barrier_type='up_and_out', rebate=5)
    print(f"Barrier Call Price: {call_price:.4f}, Barrier Put Price: {put_price:.4f}")
    call_price, put_price = pricer.monte_carlo_asian_price(n_pths=10000, n_steps=50, method='exact')
    print(f"Asian Call Price: {call_price:.4f}, Asian Put Price: {put_price:.4f}")
    call_price, put_price = pricer.monte_carlo_lookback_price(n_pths=10000, n_steps=50, method='exact')
    print(f"Lookback Call Price: {call_price:.4f}, Lookback Put Price: {put_price:.4f}")
    # call_price, put_price = pricer.monte_carlo_futures_price(n_pths=10000, n_steps=50, method='exact')
    # print(f"Futures Call Price: {call_price:.4f}, Futures Put Price: {put_price:.4f}")
    
