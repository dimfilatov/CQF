"""
Simple Monte-Carlo pricing of a zero-coupon bond and of a European call
option on a bond, following the two algorithms:

1. Bond Pricing in Practice (Monte Carlo on the short rate r):
     r(0) = r
     r_j(t_i) = r_j(t_{i-1})
                + (mu(t_{i-1}, r_j(t_{i-1})) - sigma(t_{i-1}, r_j(t_{i-1})) * theta(t_{i-1})) * (t_i - t_{i-1})
                + sigma(t_{i-1}, r_j(t_{i-1})) * Z_{j,i} * sqrt(t_i - t_{i-1})
     Int_{j,i} = Int_{j,i-1} + 0.5 * (r_j(t_i) + r_j(t_{i-1})) * (t_i - t_{i-1})
     B_j = exp(-Int_{j,M})
     B = (1/N) * sum_j B_j

2. Call Pricing in Practice (Monte Carlo on the bond, with
   F(t,U,V) = B(t,V) / B(t,U)):
     F_j = F(0,U,V) * exp(-sigma_{U,V}(0,U) * W_j - 0.5 * sigma_{U,V}(0,U)^2)
     C = (1/N) * B(0,U) * sum_j (F_j - K)^+
"""

import numpy as np


class BondOptionPricer:
    """Monte-Carlo pricing of a zero-coupon bond under a short-rate model,
    and of a European call option on a bond using the forward-bond measure.
    """

    def __init__(self, r0, mu, sigma, theta, seed=None):
        """
        r0    : initial short rate r(0)
        mu    : drift function mu(t, r)
        sigma : volatility function sigma(t, r)
        theta : deterministic function theta(t), the drift-adjustment term
                from the Monte-Carlo scheme (i.e. the measure-change /
                bond-volatility term tied to the simulation measure).
                NOTE: this is unrelated to the Hull-White calibration
                function of the same name, theta_HW(t), used inside mu(t,r)
                to fit the initial term structure.
        seed  : optional RNG seed for reproducibility
        """
        self.r0 = r0
        self.mu = mu
        self.sigma = sigma
        self.theta = theta
        self.rng = np.random.default_rng(seed)

    def simulate_short_rate_paths(self, T, N=10, M=100):
        """Simulate N short-rate paths r_j(t_i) over [0, T] with M time steps.

        Returns (t_grid, paths):
          t_grid : array of shape (M+1,) with the simulation times t_0=0,...,t_M=T
          paths  : array of shape (N, M+1) with paths[j, i] = r_j(t_i)
        """
        dt = T / M
        sqrt_dt = np.sqrt(dt)
        t_grid = np.linspace(0.0, T, M + 1)

        paths = np.empty((N, M + 1))
        paths[:, 0] = self.r0

        for j in range(N):
            r = self.r0
            t = 0.0
            for i in range(M):
                mu_i = self.mu(t, r)
                sigma_i = self.sigma(t, r)
                theta_i = self.theta(t)
                Z = self.rng.standard_normal()

                r = r + (mu_i - sigma_i * theta_i) * dt + sigma_i * Z * sqrt_dt
                t += dt
                paths[j, i + 1] = r

        return t_grid, paths

    def price_bond(self, T, N=10_000, M=100):
        """Monte-Carlo price B(0,T) of a zero-coupon bond via short-rate simulation."""
        dt = T / M
        _, paths = self.simulate_short_rate_paths(T, N=N, M=M)

        # Composite trapezoidal rule for Int_{j,M} = integral_0^T r_j(s) ds,
        # equivalent to the incremental Int_{j,i} update in the slide.
        integrals = dt * (paths[:, 0] / 2 + paths[:, 1:-1].sum(axis=1) + paths[:, -1] / 2)
        bond_payoffs = np.exp(-integrals)

        return bond_payoffs.mean()

    def plot_short_rate_paths(self, T, N=10, M=100, ax=None, show=True):
        """Simulate and display N short-rate paths r_j(t) over [0, T]."""
        import matplotlib.pyplot as plt

        t_grid, paths = self.simulate_short_rate_paths(T, N=N, M=M)

        if ax is None:
            _, ax = plt.subplots(figsize=(8, 5))

        for j in range(N):
            ax.plot(t_grid, paths[j], linewidth=1)

        ax.set_xlabel("t")
        ax.set_ylabel("r(t)")
        ax.set_title(f"Simulated short-rate paths (N={N})")

        if show:
            plt.tight_layout()
            plt.show()

        return ax

    def price_call_on_bond(self, F0_UV, B0_U, sigma_UV, K, N=10_000):
        """Monte-Carlo price of a European call on a bond maturing at V, with
        the option maturing at U < V.

        F0_UV    : forward bond price F(0,U,V) = B(0,V) / B(0,U), given by the market
        B0_U     : discount factor B(0,U), given by the market
        sigma_UV : volatility sigma_{U,V}(0,U) of the forward bond
        K        : strike
        N        : number of simulations
        """
        W = self.rng.standard_normal(N)
        F = F0_UV * np.exp(-sigma_UV * W - 0.5 * sigma_UV ** 2)
        payoff = np.maximum(F - K, 0.0)
        return B0_U * payoff.mean()


if __name__ == "__main__":
    # Simple, hand-picked (deterministic) model functions: Vasicek-style
    # mean-reverting short rate (constant long-run mean b, NOT Hull-White,
    # which would instead use a time-dependent theta_HW(t) fit to the market
    # curve: mu(t, r) = theta_HW(t) - a * r).
    def mu(t, r):
        a, b = 0.1, 0.03  # mean-reversion speed, constant long-run mean
        return a * (b - r)

    def sigma(t):
        return 0.01  # constant short-rate volatility

    def theta(t):
        # Drift-adjustment term from the Monte-Carlo scheme (measure-change /
        # bond-volatility term), NOT the Hull-White calibration function.
        return 0.02  # constant, for simplicity

    r0 = 0.03
    pricer = BondOptionPricer(
        r0=r0,
        mu=mu,
        sigma=lambda t, r: sigma(t),
        theta=theta,
        seed=42,
    )

    # --- 1. Bond pricing via Monte Carlo on the short rate ---
    T = 1.0
    B0_U = pricer.price_bond(T=T, N=5_000, M=252)
    print(f"Zero-coupon bond price B(0,{T}) = {B0_U:.4f}")

    # --- Display a handful of simulated short-rate paths ---
    pricer.plot_short_rate_paths(T=T, N=20, M=252)

    # --- 2. Call option on a bond via Monte Carlo on the forward bond ---
    # Market-given inputs for the call (U = option maturity, V = bond maturity > U).
    F0_UV = 0.90      # F(0,U,V) = B(0,V)/B(0,U), given by the market
    sigma_UV = 0.02    # sigma_{U,V}(0,U), set here as a constant
    K = 0.88           # strike

    C = pricer.price_call_on_bond(F0_UV=F0_UV, B0_U=B0_U, sigma_UV=sigma_UV, K=K, N=5_000)
    print(f"Call option price C = {C:.4f}")
