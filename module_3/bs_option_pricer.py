import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import norm
from scipy.stats import qmc, norm   
from quantmod.markets import getTicker
import quantmod.charts
from datetime import datetime

class BlackScholesOptionPricer:
    def __init__(self, S, K, T, t, r, sigma, dividend):
        self.S = S  # Current stock price
        self.K = K  # Strike price
        self.T = T  # Time to maturity
        self.t = t  # Current time
        self.tau = self.T - self.t  # Time to maturity
        self.r = r  # Risk-free rate
        self.dividend = dividend  # Dividend yield
        self.sigma = sigma  # Volatility
        self.d1 = (np.log(self.S/self.K) + (self.r - self.dividend + 0.5 * self.sigma**2) * self.tau) / (self.sigma * np.sqrt(self.tau))
        self.d2 = self.d1 - self.sigma * np.sqrt(self.tau)

    def simulate_GBM_paths(self, n_pths, n_steps, method):
        dt = self.tau / n_steps
        paths = np.zeros((n_pths, n_steps+1))
        paths[:, 0] = self.S

        if method == 'naive':
            for t in range(1, n_steps+1):
                z = np.random.standard_normal(n_pths)
                paths[:, t] = paths[:, t-1] * (1 + (self.r - self.dividend) * dt + self.sigma * np.sqrt(dt) * z)
        elif method == 'exact':
            for t in range(1, n_steps + 1):
                z = np.random.standard_normal(n_pths)
                paths[:, t] = paths[:, t-1] * np.exp((self.r - self.dividend - 0.5 * self.sigma **2) * dt + self.sigma * np.sqrt(dt) * z)
        elif method == 'moment_matcing':
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
            sobol = qmc.Sobol(d=1, scramble=True)
            u = sobol.random(n_pths)
            z = norm.ppf(u.ravel())  # Inverse transform to standard normal
            S_T = self.S * np.exp((self.r - self.dividend - 0.5 * self.sigma**2) * self.tau + self.sigma * np.sqrt(self.tau) * z)

        if method not in ['moment_matcing', 'antithetic', 'sobol']:
            S_T = paths[:, -1]          
        
        return S_T, paths
    
    def analytical_price(self):
        call_price = np.exp(-self.dividend * (self.tau)) * self.S * norm.cdf(self.d1) - self.K * np.exp(- self.r * (self.tau)) * norm.cdf(self.d2)
        put_price = self.K * np.exp(-self.r * (self.tau)) * norm.cdf(-self.d2) - np.exp(-self.dividend * (self.tau)) * self.S * norm.cdf(-self.d1)
        return call_price, put_price
    
    def monte_carlo_european_price(self, n_pths, n_steps, method):
        S_T, paths = self.simulate_GBM_paths(n_pths, n_steps, method)
        call_payoffs = np.maximum(S_T - self.K, 0)
        put_payoffs = np.maximum(self.K - S_T, 0)
        call_price = np.exp(- self.r * self.tau) * np.mean(call_payoffs)
        put_price = np.exp(- self.r * self.tau) * np.mean(put_payoffs)
        return call_price, put_price

    def monte_carlo_asian_price(self, n_pths, n_steps, method):
        S_T, paths = self.simulate_GBM_paths(n_pths, n_steps, method)
        S_avg = np.mean(paths, axis = 1)
        payoffs_call = np.maximum(S_avg - self.K, 0)
        payoffs_put = np.maximum(self.K - S_avg, 0)
        call_price = np.exp(-self.r * self.tau) * np.mean(payoffs_call)
        put_price = np.exp(-self.r * self.tau) * np.mean(payoffs_put)
        return call_price, put_price
    
    def monte_carlo_barrier_price(self, n_pths, n_steps, method, barrier, barrier_type, rebate):
        S_T, paths = self.simulate_GBM_paths(n_pths, n_steps, method)
        if barrier_type == "up_and_out":
            valid_paths = ~np.any(paths > barrier, axis=1)
        elif barrier_type == "down_and_out":
            valid_paths = ~np.any(paths < barrier, axis=1)
        elif barrier_type == "up_and_in":
            breach = np.any(paths > barrier, axis=1)
        elif barrier_type == "down_and_in":
            valid_paths = np.any(paths < barrier, axis = 1)
        else:            
            raise ValueError("Invalid barrier type")
        
        payoffs_call =np.where(valid_paths, np.maximum(S_T - self.K, 0), rebate)
        payoffs_put = np.where(valid_paths, np.maximum(self.K - S_T, 0), rebate)
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
        methods = ['naive', 'exact', 'moment_matcing', 'antithetic', 'sobol']
        results = []
        for method in methods:
            call_price, put_price = self.monte_carlo_european_price(n_pths, n_steps, method)
            results.append((method, call_price, put_price))
        analytical_call, analytical_put = self.analytical_price()
        results.append(('analytical', analytical_call, analytical_put))
        df = pd.DataFrame(results, columns=['Method', 'Call Price', 'Put Price'])
        print(df)

    def calculate_greeks_for_european(self, option_type):
        if option_type == 'call':
            delta = np.exp(-self.dividend * self.tau) * norm.cdf(self.d1)
            theta = (-self.S * norm.pdf(self.d1) * self.sigma / (2 * np.sqrt(self.tau)) - self.r * self.K * np.exp(-self.r * self.tau) * norm.cdf(self.d2)) / 365
            rho = self.K * self.tau * np.exp(-self.r * self.tau) * norm.cdf(self.d2) / 100
        elif option_type == "put":
            delta = np.exp(-self.dividend * self.tau) * (norm.cdf(self.d1) - 1)
            theta = (-self.S * norm.pdf(self.d1) * self.sigma / (2 * np.sqrt(self.tau)) + self.r * self.K * np.exp(-self.r * self.tau) * norm.cdf(-self.d2)) / 365
            rho = -self.K * self.tau * np.exp(-self.r * self.tau) * norm.cdf(-self.d2) / 100
        else:
            raise ValueError("Invalid option type")
        gamma = np.exp(-self.dividend * self.tau) * norm.pdf(self.d1) / (self.sigma * self.S * np.sqrt(self.tau)) 
        vega = self.S * np.exp(-self.dividend * self.tau) * norm.pdf(self.d1) * np.sqrt(self.tau)
        return delta, gamma, vega, theta, rho

def compute_IV_from_price(option_price, spot_price, option_type, K, T, t, r, dividend, tol=1e-6, max_iter=100):
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

def calculate_greeks_for_market_data(spot_price, ticker, maturity, strike_range, option_type, r, dividend):
    spy = getTicker(ticker)
    options = spy.option_chain(maturity)
    dte = (datetime.strptime(maturity, '%Y-%m-%d') - datetime.today()).days/365
    if option_type == 'call':
        df = options.calls[(options.calls['strike'] >= strike_range[0]) & (options.calls['strike'] <= strike_range[1]   )].copy()
    else:
        df = options.puts[(options.puts['strike'] >= strike_range[0]) & (options.puts['strike'] <= strike_range[1]   )].copy()
    df = df[['strike', 'lastPrice']]
    # df.reset_index(drop=True, inplace=True)

    for idx, row in df.iterrows():
        implied_vol =compute_IV_from_price(row['lastPrice'], spot_price, option_type, row['strike'], dte, 0, r, dividend)
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
    plt.subplot(2, 3, 6)
    plt.plot(df.index, df['Implied Vol'], marker='o')
    plt.title('Implied Vol vs Strike')
    plt.xlabel('Strike Price')
    plt.ylabel('Implied Volatility')
    plt.grid()
    plt.tight_layout()
    plt.show() 


if __name__ == "__main__":
    S = 100
    K = 100
    T = 1
    t = 0
    r = 0.05
    sigma = 0.2
    dividend = 0.02
    pricer = BlackScholesOptionPricer(S, K, T, t, r, sigma, dividend)
    pricer.compare_methods_against_analytical(n_pths=10000, n_steps=50)
    calculate_greeks_for_market_data(spot_price=658, ticker="SPY", maturity="2026-03-30", strike_range=(650, 700), option_type='call', r=0.05, dividend=0.02)