"""Price simulation utilities for use in Jupyter notebooks.

Provides a `PriceSimulator` class that can simulate Geometric Brownian Motion
price paths and plot them (matplotlib and optional Plotly output).

Example:
    from price_simulator import PriceSimulator
    sim = PriceSimulator(S0=100)
    paths = sim.simulate_gbm(steps=252, paths=10)
    sim.plot(paths)

"""

import numpy as np
from typing import Optional
import matplotlib.pyplot as plt

class StochasticProcessSimulator:
    """Simulate and plot asset price paths.

    Parameters
    ----------
    S0 : float
        Initial price.
    mu : float
        Drift (annualized).
    sigma : float
        Volatility (annualized).
    dt : float
        Time step in years (default is daily steps: 1/252).
    seed : Optional[int]
        RNG seed for reproducibility.
    """

    def __init__(self, S0: float = 100.0, mu: float = 0.05, sigma: float = 0.2, dt: float = 1/252, seed: Optional[int] = None):
        self.S0 = float(S0)
        self.mu = float(mu)
        self.sigma = float(sigma)
        self.dt = float(dt)
        self._rng = np.random.default_rng(seed)

    def simulate_gbm(self, steps: int = 252, paths: int = 1, method: str = 'exact') -> np.ndarray:
        """Simulate Geometric Brownian Motion paths.

        Parameters
        ----------
        steps, paths: as before
        method: 'exact' uses the closed-form log-normal increments; 'euler' uses Euler-Maruyama on S.

        Returns an array with shape (steps+1, paths).
        """
        if method not in ('exact', 'euler'):
            raise ValueError("method must be 'exact' or 'euler'")
        if method == 'exact':
            return self.simulate_gbm_exact(steps=steps, paths=paths)
        return self.simulate_gbm_euler(steps=steps, paths=paths)

    def simulate_gbm_exact(self, steps: int = 252, paths: int = 1) -> np.ndarray:
        """Exact-distribution simulation for GBM using log-normal increments.

        Returns shape (steps+1, paths).
        """
        steps = int(steps)
        paths = int(paths)
        drift = (self.mu - 0.5 * self.sigma ** 2) * self.dt
        vol = self.sigma * np.sqrt(self.dt)

        increments = np.random.normal(size=(steps, paths))
        log_returns = drift + vol * increments
        log_paths = np.vstack([np.zeros((1, paths)), np.cumsum(log_returns, axis=0)])
        prices = self.S0 * np.exp(log_paths)
        return prices

    def simulate_gbm_euler(self, steps: int = 252, paths: int = 1) -> np.ndarray:
        """Euler-Maruyama simulation for GBM on S:

        S_{t+1} = S_t + mu * S_t * dt + sigma * S_t * dW
        Returns shape (steps+1, paths).
        """
        steps = int(steps)
        paths = int(paths)
        sqrt_dt = np.sqrt(self.dt)

        S = np.empty((steps + 1, paths), dtype=float)
        S[0, :] = self.S0
        increments = np.random.normal(size=(steps, paths))
        for t in range(steps):
            dW = increments[t] * sqrt_dt
            S[t + 1] = S[t] + self.mu * S[t] * self.dt + self.sigma * S[t] * dW
        return S

    def simulate_ou(self, steps: int = 252, paths: int = 1, theta: float = 1.0, mu_level: float = 0.0, sigma: float = 0.3, X0: Optional[float] = None) -> np.ndarray:
        """Simulate Ornstein-Uhlenbeck process.

        dX = theta*(mu_level - X) dt + sigma dW
        Only uses approximate Euler-Maruyama.
        Returns array shape (steps+1, paths).
        """
        steps = int(steps)
        paths = int(paths)
        if X0 is None:
            X0 = mu_level

        X = np.empty((steps + 1, paths), dtype=float)
        X[0, :] = float(X0)

        normals = np.random.normal(size=(steps, paths))
        sqrt_dt = np.sqrt(self.dt)
        for t in range(steps):
            dW = normals[t] * sqrt_dt
            X[t + 1] = X[t] + theta * (mu_level - X[t]) * self.dt + sigma * dW

        return X

    def simulate_cox_ingersoll_ross(self, steps, paths, theta, mu_level, sigma, X0=None):
        """Simulate Cox-Ingersoll-Ross (CIR) process.

        dX = theta*(mu_level - X) dt + sigma*sqrt(X) dW
        Only uses approximate Euler-Maruyama.
        Returns array shape (steps+1, paths).
        """
        steps = int(steps)
        paths = int(paths)
        if X0 is None:
            X0 = mu_level

        X = np.empty((steps + 1, paths), dtype=float)
        X[0, :] = float(X0)

        normals = np.random.normal(size=(steps, paths))
        sqrt_dt = np.sqrt(self.dt)
        for t in range(steps):
            dW = normals[t] * sqrt_dt
            sqrt_X_t = np.sqrt(np.maximum(X[t], 0))  # Ensure non-negativity for sqrt
            X[t + 1] = X[t] + theta * (mu_level - X[t]) * self.dt + sigma * sqrt_X_t * dW
            X[t + 1] = np.maximum(X[t + 1], 0)  # Ensure non-negativity

        return X

    def simulate_correlated_euler(self, steps: int = 252, n_assets: int = 2, n_paths: int = 1, mu: Optional[np.ndarray] = None, sigma: Optional[np.ndarray] = None, corr: Optional[np.ndarray] = None, S0: Optional[np.ndarray] = None) -> np.ndarray:
        """Simulate multiple correlated asset prices using Euler-Maruyama.

        Parameters
        - n_assets: number of assets
        - n_paths: number of simulated paths per asset
        - mu: array-like of shape (n_assets,) of drifts (annualized). If None uses self.mu
        - sigma: array-like of vols (n_assets,). If None uses self.sigma for all.
        - corr: correlation matrix (n_assets x n_assets). If None, identity is used.
        - S0: initial prices (n_assets,) or scalar. If None, uses self.S0.

        Returns an array of shape (steps+1, n_assets, n_paths).
        """
        steps = int(steps)
        n_assets = int(n_assets)
        n_paths = int(n_paths)
        dt = self.dt
        sqrt_dt = np.sqrt(dt)

        if mu is None:
            mu = np.full(n_assets, self.mu, dtype=float)
        else:
            mu = np.asarray(mu, dtype=float)
        if sigma is None:
            sigma = np.full(n_assets, self.sigma, dtype=float)
        else:
            sigma = np.asarray(sigma, dtype=float)
        if S0 is None:
            S0 = np.full(n_assets, self.S0, dtype=float)
        else:
            S0 = np.asarray(S0, dtype=float)

        if corr is None:
            corr = np.eye(n_assets)
        else:
            corr = np.asarray(corr, dtype=float)

        L = np.linalg.cholesky(corr)

        S = np.empty((steps + 1, n_assets, n_paths), dtype=float)
        S[0, :, :] = np.expand_dims(S0, axis=(1))

        normals = np.random.normal(size=(steps, n_assets, n_paths))
        for t in range(steps):
            z = normals[t]  # shape (n_assets, n_paths)
            correlated = L @ z  # shape (n_assets, n_paths)
            dW = correlated * sqrt_dt
            drift_term = (mu[:, None] * S[t]) * dt
            diffusion_term = (sigma[:, None] * S[t]) * dW
            S[t + 1] = S[t] + drift_term + diffusion_term

        return S

    def plot(self, prices: np.ndarray, title: Optional[str] = None, figsize: tuple = (10, 6), show_legend: bool = False):
        """Plot price paths using matplotlib.

        `prices` should be an array of shape (steps+1, paths).
        Returns the matplotlib figure object.
        """
        if prices.ndim == 1:
            prices = prices.reshape(-1, 1)

        steps_plus_one, n_paths = prices.shape
        fig, ax = plt.subplots(figsize=figsize)
        for i in range(n_paths):
            ax.plot(np.arange(steps_plus_one), prices[:, i], lw=1)

        ax.set_xlabel('Step')
        ax.set_ylabel('Price')
        ax.set_title(title or 'Simulated Price Paths')
        if show_legend and n_paths <= 20:
            ax.legend([f'Path {i+1}' for i in range(n_paths)])
        ax.grid(True)
        plt.tight_layout()
        plt.show()
        return fig

if __name__ == '__main__':
    # quick demo when run as script
    sim = StochasticProcessSimulator(S0=100, mu=0.05, sigma=0.2, seed=42)
    p = sim.simulate_gbm(steps=252, paths=50, method='exact')
    sim.plot(p, title='Demo: GBM (50 paths)')

    # quick demo when run as script
    p_ou = sim.simulate_ou(steps=252, paths=10, theta=10.0, mu_level=0.05, sigma=0.03, X0=0.1)
    p_cir = sim.simulate_cox_ingersoll_ross(steps=252, paths=10, theta=10.0, mu_level=0.05, sigma=0.03, X0=0.1)
    sim.plot(p_ou, title='Demo: OU (10 paths)')
    sim.plot(p_cir, title='Demo: CIR (10 paths)')

    # quick demo when run as script
    p = sim.simulate_correlated_euler(steps=252, n_assets=2, n_paths=6, mu=[0.05, 0.03], sigma=[0.2, 0.1], corr=[[1.0, 0.8], [0.8, 1.0]], S0=[100, 50])
    sim.plot(p[:, 0, :], title='Demo: Correlated GBM (Asset 1)')
    sim.plot(p[:, 1, :], title='Demo: Correlated GBM (Asset 2)')