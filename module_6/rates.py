"""
HJM (Heath-Jarrow-Morton) forward-rate model calibrated via Principal
Component Analysis (PCA) on historical forward-rate curves, under the
Musiela parametrisation (tau = T - t = time-to-maturity).

Workflow:
  1. Load historical forward curves F(t,T) from ./data/rates.csv.
  2. Compute the daily changes dr_t = F(t+1,.) - F(t,.).
  3. Build the (annualised) covariance matrix of dr_t.
  4. Eigen-decompose the covariance matrix.
  5. Keep the three largest eigenvalues/eigenvectors (the three main
     factors: level, slope, curvature).
  6. Build the volatility functions (principal components) as
     vol_i(tau) = eigenvector_i(tau) * sqrt(eigenvalue_i).
  7. Fit PC1 as a flat function (median across tenors) and PC2, PC3 as
     cubic polynomials in tau.
  8. Simulate forward curves forward in time using the no-arbitrage HJM
     drift under the Musiela parametrisation:

        df(t,tau) = ( sum_i v_i(t,tau) * Int_0^tau v_i(t,s) ds ) dt
                    + sum_i v_i(t,tau) dX_i
                    + df/dtau dt

     where dX_i = phi_i * sqrt(dt), phi_i ~ N(0,1) i.i.d. standard normal.
"""

import numpy as np
import pandas as pd
from scipy.integrate import cumulative_trapezoid


class Rates:
    """Historical forward-rate curves, PCA-based HJM volatility calibration,
    and Monte-Carlo simulation of forward rates under the Musiela
    parametrisation.
    """

    def __init__(self, filepath="./data/rates.csv", dt=1 / 252, n_factors=3):
        """
        filepath   : path to the CSV file of historical forward curves.
                     First column = day index (business days). First row
                     (header) = tenors tau (in years) for the remaining
                     columns. Each cell = F(t,T) in percent.
        dt         : time step used to annualise daily changes and used as
                     the simulation time step (default 1/252 = one trading
                     day).
        n_factors  : number of principal components (HJM factors) to keep.
        """
        self.filepath = filepath
        self.dt = dt
        self.n_factors = n_factors

        self.days = None            # (n_days,) day index of each observation
        self.tenors = None          # (n_tenors,) tenor grid tau, in years
        self.curves = None          # (n_days, n_tenors) F(t,T), in decimal

        self.dr = None              # (n_days-1, n_tenors) daily changes dr_t
        self.cov_matrix = None      # (n_tenors, n_tenors) annualised covariance

        self.eigenvalues = None     # (n_tenors,) sorted descending
        self.eigenvectors = None    # (n_tenors, n_tenors) matching columns

        self.top_eigenvalues = None   # (n_factors,)
        self.top_eigenvectors = None  # (n_tenors, n_factors)

        self.principal_components = None  # (n_tenors, n_factors) vol_i(tau)
        self.pc_fits = None                # (n_tenors, n_factors) fitted vol_i(tau)
        self.pc2_coeffs = None             # cubic fit coefficients for PC2
        self.pc3_coeffs = None             # cubic fit coefficients for PC3
        self.pc2_r2 = None                 # R^2 of the PC2 cubic fit
        self.pc3_r2 = None                 # R^2 of the PC3 cubic fit

    # ------------------------------------------------------------------
    # 2. Load data
    # ------------------------------------------------------------------
    def load_data(self):
        """Load F(t,T) from ``self.filepath``.

        The first column holds the day index; the header row (other than
        the first column) holds the tenors tau. Rates are stored in the
        CSV as percent and are converted here to decimal.
        """
        df = pd.read_csv(self.filepath)

        self.days = df.iloc[:, 0].to_numpy()
        self.tenors = df.columns[1:].astype(float).to_numpy()
        self.curves = df.iloc[:, 1:].to_numpy(dtype=float) / 100.0

        return self.curves

    def get_curve(self, day_index=-1):
        """Return the forward curve F(t,.) observed on ``day_index`` (by
        default the most recent available day, i.e. "today")."""
        return self.curves[day_index, :]

    # ------------------------------------------------------------------
    # 4. dr_t
    # ------------------------------------------------------------------
    def compute_dr(self):
        """Compute the historical daily changes dr_t = F(t+1,T) - F(t,T)
        for every tenor T."""
        self.dr = np.diff(self.curves, axis=0)
        return self.dr

    # ------------------------------------------------------------------
    # 5. Covariance matrix of dr_t
    # ------------------------------------------------------------------
    def compute_covariance_matrix(self):
        """Compute the (annualised) covariance matrix of dr_t, i.e. divide
        the sample covariance of the daily changes by dt."""
        self.cov_matrix = np.cov(self.dr, rowvar=False) / self.dt
        return self.cov_matrix

    # ------------------------------------------------------------------
    # 6. Eigenvalues / eigenvectors
    # ------------------------------------------------------------------
    def compute_eigen(self):
        """Eigen-decompose the covariance matrix and sort the
        eigenvalues/eigenvectors in descending order of eigenvalue."""
        # cov_matrix is symmetric (real) -> eigh is faster/more accurate than eig.
        eigvals, eigvecs = np.linalg.eigh(self.cov_matrix)

        order = np.argsort(eigvals)[::-1]
        self.eigenvalues = eigvals[order]
        self.eigenvectors = eigvecs[:, order]

        return self.eigenvalues, self.eigenvectors

    # ------------------------------------------------------------------
    # 7. Top eigenvalues/eigenvectors
    # ------------------------------------------------------------------
    def select_top_eigen(self, n_factors=None):
        """Keep the ``n_factors`` largest eigenvalues and their associated
        eigenvectors (default: ``self.n_factors``, i.e. 3)."""
        n_factors = n_factors or self.n_factors
        self.n_factors = n_factors

        self.top_eigenvalues = self.eigenvalues[:n_factors]
        self.top_eigenvectors = self.eigenvectors[:, :n_factors].copy()

        # Fix the sign ambiguity of eigenvectors: make each eigenvector
        # predominantly positive so that the resulting vol functions have a
        # consistent, interpretable sign (common convention for PCA on
        # yield-curve data).
        for i in range(n_factors):
            if np.mean(self.top_eigenvectors[:, i]) < 0:
                self.top_eigenvectors[:, i] *= -1

        return self.top_eigenvalues, self.top_eigenvectors

    # ------------------------------------------------------------------
    # 8. Principal components = volatility functions
    # ------------------------------------------------------------------
    def compute_principal_components(self):
        """Build the volatility functions vol_i(tau) = eigenvector_i(tau) *
        sqrt(eigenvalue_i) for each of the ``n_factors`` kept factors."""
        self.principal_components = self.top_eigenvectors * np.sqrt(self.top_eigenvalues)
        return self.principal_components

    # ------------------------------------------------------------------
    # 9. Fit PC1 as a flat (median) function of tau
    # ------------------------------------------------------------------
    def fit_pc1(self):
        """Fit PC1 (the "level" factor) as a constant equal to the median
        of PC1 across all tenors."""
        pc1 = self.principal_components[:, 0]
        level = np.median(pc1)
        fitted = np.full_like(pc1, level)

        self._store_fit(0, fitted)
        return fitted

    # ------------------------------------------------------------------
    # 10. Fit PC2, PC3 as cubic polynomials in tau
    # ------------------------------------------------------------------
    def _fit_cubic(self, pc):
        """Least-squares fit of ``pc`` as a cubic polynomial in tau:
        a0 + a1*tau + a2*tau^2 + a3*tau^3.

        Returns (coeffs, fitted_values, r_squared), where r_squared is the
        coefficient of determination R^2 = 1 - SS_res/SS_tot of the fit."""
        tau = self.tenors
        X = np.column_stack([np.ones_like(tau), tau, tau ** 2, tau ** 3])
        coeffs, *_ = np.linalg.lstsq(X, pc, rcond=None)
        fitted = X @ coeffs

        ss_res = np.sum((pc - fitted) ** 2)
        ss_tot = np.sum((pc - np.mean(pc)) ** 2)
        r_squared = 1.0 - ss_res / ss_tot

        return coeffs, fitted, r_squared

    def fit_pc2(self):
        """Fit PC2 (the "slope" factor) as a cubic polynomial in tau."""
        self.pc2_coeffs, fitted, self.pc2_r2 = self._fit_cubic(self.principal_components[:, 1])
        self._store_fit(1, fitted)
        return fitted

    def fit_pc3(self):
        """Fit PC3 (the "curvature" factor) as a cubic polynomial in tau."""
        self.pc3_coeffs, fitted, self.pc3_r2 = self._fit_cubic(self.principal_components[:, 2])
        self._store_fit(2, fitted)
        return fitted

    def _store_fit(self, index, fitted):
        if self.pc_fits is None:
            self.pc_fits = np.full_like(self.principal_components, np.nan)
        self.pc_fits[:, index] = fitted

    def fit_volatility_functions(self):
        """Fit all ``n_factors`` volatility functions (PC1 flat, PC2 and
        PC3 cubic) and return the fitted vol_i(tau) matrix, shape
        (n_tenors, n_factors)."""
        self.fit_pc1()
        self.fit_pc2()
        self.fit_pc3()
        return self.pc_fits

    # ------------------------------------------------------------------
    # 11. HJM Monte-Carlo simulation (Musiela parametrisation)
    # ------------------------------------------------------------------
    def _hjm_drift(self):
        """Compute m(tau) = sum_i vol_i(tau) * Int_0^tau vol_i(s) ds for the
        fitted volatility functions (time-homogeneous drift term)."""
        tau = self.tenors
        m = np.zeros_like(tau)
        for i in range(self.pc_fits.shape[1]):
            vol_i = self.pc_fits[:, i]
            # Int_0^tau vol_i(s) ds, cumulative from tau=0.
            integral_i = cumulative_trapezoid(vol_i, tau, initial=0.0)
            m += vol_i * integral_i
        return m

    def simulate(self, n_steps=1, n_paths=1, seed=None):
        """Simulate forward curves forward in time using the Musiela HJM SDE:

            f(t+dt, tau) = f(t, tau) + dfbar(t, tau)

            dfbar(t, tau) = m(tau)*dt
                            + sum_i vol_i(tau) * phi_i * sqrt(dt)
                            + (df/dtau)(t, tau) * dt

        where phi_i ~ N(0,1) i.i.d. standard normal are drawn independently
        for each factor i, each time step and each path, m(tau) is the
        no-arbitrage HJM drift and df/dtau is the (finite-difference)
        derivative of the simulated curve along the tenor axis.

        Returns an array of shape (n_paths, n_steps+1, n_tenors) with the
        simulated forward curves, where index 0 along the time axis is
        today's curve.
        """
        rng = np.random.default_rng(seed)

        tau = self.tenors
        dt = self.dt
        sqrt_dt = np.sqrt(dt)

        vols = self.pc_fits                 # (n_tenors, n_factors)
        drift = self._hjm_drift()           # (n_tenors,)
        f0 = self.get_curve(day_index=-1)   # today's curve, "day 0"

        paths = np.empty((n_paths, n_steps + 1, tau.size))
        paths[:, 0, :] = f0

        for p in range(n_paths):
            f = f0.copy()
            for step in range(n_steps):
                # d f / d tau, via (non-uniform) finite differences.
                df_dtau = np.gradient(f, tau)

                phi = rng.standard_normal(self.n_factors)
                diffusion = vols @ (phi * sqrt_dt)

                f = f + drift * dt + diffusion + df_dtau * dt
                paths[p, step + 1, :] = f

        return paths

    def get_rate(self, curve, tenor):
        """Interpolate/read off the forward rate for ``tenor`` (in years)
        from a forward curve array aligned with ``self.tenors``."""
        return np.interp(tenor, self.tenors, curve)

    # ------------------------------------------------------------------
    # Zero-coupon bonds / yields
    # ------------------------------------------------------------------
    def _tenor_bounds(self):
        """Tenor grid including T_{-1}=0 (length n_tenors+1) and the bucket
        widths T_i - T_{i-1} (length n_tenors), aligned with ``self.tenors``
        (each tenors[i] is read as T_i, with T_{-1}=0)."""
        T = np.concatenate(([0.0], self.tenors))
        dT = np.diff(T)
        return T, dT

    def zcb(self, curve=None, day_index=-1):
        """Compute zero-coupon bond prices Z(0,T_i) recursively from the
        instantaneous forward rates f(T_{i-1},T_i), using

            -ln(Z(0,T_i)/Z(0,T_{i-1})) / (T_i - T_{i-1}) = f(T_{i-1},T_i)
            Z(0,0) = 1

        i.e. Z(0,T_i) = Z(0,T_{i-1}) * exp(-f(T_{i-1},T_i)*(T_i-T_{i-1})).

        ``curve`` is a forward-rate array aligned with ``self.tenors`` (e.g.
        f(0,0.08), f(0.08,0.5), ...); if not given, the historical forward
        curve at ``day_index`` (default: most recent, "today") is used.

        Returns (T_grid, Z), where T_grid = [0, tenors...] (length
        n_tenors+1) and Z matches it, with Z[0] = Z(0,0) = 1.
        """
        if curve is None:
            curve = self.get_curve(day_index=day_index)

        T, dT = self._tenor_bounds()
        log_Z = -np.cumsum(curve * dT)
        Z = np.concatenate(([1.0], np.exp(log_Z)))
        return T, Z

    def compute_yield(self, curve=None, day_index=-1):
        """Compute the (continuously-compounded) yield curve
        y(0,T) = -ln(Z(0,T)) / T from the ZCB prices computed by ``zcb()``.

        Returns an array aligned with ``self.tenors`` (T=0 is excluded,
        since the yield formula is undefined there).
        """
        T, Z = self.zcb(curve=curve, day_index=day_index)
        return -np.log(Z[1:]) / T[1:]

    def forward_zcb(self, curve=None, day_index=-1):
        """Compute forward zero-coupon bond prices
        Z(0; T_{i-1},T_i) = exp(-f(T_{i-1},T_i) * (T_i - T_{i-1}))
        for each tenor bucket, aligned with ``self.tenors``."""
        if curve is None:
            curve = self.get_curve(day_index=day_index)

        _, dT = self._tenor_bounds()
        return np.exp(-curve * dT)

    def verify_forward_zcb(self, curve=None, day_index=-1, tol=1e-10):
        """Sanity check that, for every tenor bucket i,
        Z(0,T_{i-1}) * Z(0; T_{i-1},T_i) = Z(0,T_i), consistently with the
        ZCB recursion used in ``zcb()``.

        Returns True if the maximum absolute discrepancy is below ``tol``;
        raises AssertionError otherwise.
        """
        T, Z = self.zcb(curve=curve, day_index=day_index)
        fwd_zcb = self.forward_zcb(curve=curve, day_index=day_index)

        lhs = Z[:-1] * fwd_zcb   # Z(0,T_{i-1}) * Z(0; T_{i-1},T_i)
        rhs = Z[1:]              # Z(0,T_i)
        max_diff = np.max(np.abs(lhs - rhs))

        assert max_diff < tol, (
            "ZCB recursion sanity check failed: max |Z(0,T_i-1)*Z(0;T_i-1,T_i) "
            f"- Z(0,T_i)| = {max_diff:.3e}"
        )
        return True

    # ------------------------------------------------------------------
    # Plotting
    # ------------------------------------------------------------------
    def plot_forward_curves(self, curves=None, labels=None, ax=None, show=True):
        """Plot one or more forward curves F(.,tau) against tenor tau.

        curves : list of 1D arrays aligned with ``self.tenors`` (defaults
                 to today's curve only).
        labels : legend labels matching ``curves``.
        """
        import matplotlib.pyplot as plt

        if curves is None:
            curves = [self.get_curve(day_index=-1)]
            labels = labels or ["Today"]

        if ax is None:
            _, ax = plt.subplots(figsize=(8, 5))

        for curve, label in zip(curves, labels):
            ax.plot(self.tenors, curve * 100, marker="o", markersize=3, label=label)

        ax.set_xlabel("Tenor tau (years)")
        ax.set_ylabel("Forward rate (%)")
        ax.set_title("Forward curves")
        ax.legend()

        if show:
            plt.tight_layout()
            plt.show()

        return ax

    def plot_zcb_curves(self, curves=None, labels=None, ax=None, show=True):
        """Plot one or more zero-coupon bond price curves Z(0,T) against
        maturity T.

        curves : list of Z arrays including the T=0 point (Z(0,0)=1), as
                 returned by ``zcb()[1]`` (defaults to today's ZCB curve).
        labels : legend labels matching ``curves``.
        """
        import matplotlib.pyplot as plt

        if curves is None:
            _, z_today = self.zcb()
            curves = [z_today]
            labels = labels or ["Today"]

        T, _ = self._tenor_bounds()

        if ax is None:
            _, ax = plt.subplots(figsize=(8, 5))

        for curve, label in zip(curves, labels):
            ax.plot(T, curve, marker="o", markersize=3, label=label)

        ax.set_xlabel("Maturity T (years)")
        ax.set_ylabel("Z(0,T)")
        ax.set_title("Zero-coupon bond price curves")
        ax.legend()

        if show:
            plt.tight_layout()
            plt.show()

        return ax

    def plot_yield_curves(self, curves=None, labels=None, ax=None, show=True):
        """Plot one or more (continuously-compounded) yield curves y(0,T)
        against maturity T.

        curves : list of yield arrays aligned with ``self.tenors``, as
                 returned by ``compute_yield()`` (defaults to today's yield
                 curve).
        labels : legend labels matching ``curves``.
        """
        import matplotlib.pyplot as plt

        if curves is None:
            curves = [self.compute_yield()]
            labels = labels or ["Today"]

        if ax is None:
            _, ax = plt.subplots(figsize=(8, 5))

        for curve, label in zip(curves, labels):
            ax.plot(self.tenors, curve * 100, marker="o", markersize=3, label=label)

        ax.set_xlabel("Maturity T (years)")
        ax.set_ylabel("Yield (%)")
        ax.set_title("Yield curves")
        ax.legend()

        if show:
            plt.tight_layout()
            plt.show()

        return ax

    def plot_simulated_rates(self, paths, tenors_to_track=(0.08, 5, 25), ax=None, show=True):
        """Plot simulated forward-rate paths f(t, tau) against simulation
        time t, for a handful of tracked tenors tau.

        paths : array returned by ``simulate()``, shape
                (n_paths, n_steps+1, n_tenors).
        """
        import matplotlib.pyplot as plt

        n_paths, n_steps_plus1, _ = paths.shape
        t_grid = np.arange(n_steps_plus1) * self.dt

        if ax is None:
            _, ax = plt.subplots(figsize=(8, 5))

        for tenor in tenors_to_track:
            idx = int(np.argmin(np.abs(self.tenors - tenor)))
            for p in range(n_paths):
                ax.plot(
                    t_grid,
                    paths[p, :, idx] * 100,
                    label=f"f(t, {self.tenors[idx]:g})" if p == 0 else None,
                )

        ax.set_xlabel("Simulation time t (years)")
        ax.set_ylabel("Forward rate (%)")
        ax.set_title("Simulated forward rates")
        ax.legend()

        if show:
            plt.tight_layout()
            plt.show()

        return ax

    def plot_covariance_matrix(self, ax=None, show=True):
        """Plot the (annualised) covariance matrix of dr_t as a heatmap
        over the tenor grid."""
        import matplotlib.pyplot as plt

        if ax is None:
            _, ax = plt.subplots(figsize=(7, 6))

        im = ax.imshow(
            self.cov_matrix,
            origin="lower",
            cmap="viridis",
            extent=[self.tenors[0], self.tenors[-1], self.tenors[0], self.tenors[-1]],
        )
        plt.colorbar(im, ax=ax, label="Covariance")

        ax.set_xlabel("Tenor tau (years)")
        ax.set_ylabel("Tenor tau (years)")
        ax.set_title("Covariance matrix of dr_t (annualised)")

        if show:
            plt.tight_layout()
            plt.show()

        return ax


if __name__ == "__main__":
    rates = Rates(filepath="./data/rates.csv", dt=1 / 252, n_factors=3)

    rates.load_data()
    rates.compute_dr()
    rates.compute_covariance_matrix()
    rates.compute_eigen()
    top_eigenvalues, _ = rates.select_top_eigen(n_factors=3)
    rates.compute_principal_components()
    rates.fit_volatility_functions()

    total_variance = rates.eigenvalues.sum()
    explained = top_eigenvalues / total_variance
    print("Top 3 eigenvalues:", np.round(top_eigenvalues, 6))
    print("Variance explained by PC1/PC2/PC3: "
          f"{explained[0]:.2%} / {explained[1]:.2%} / {explained[2]:.2%} "
          f"(cumulative {explained.sum():.2%})")

    print(f"\nCubic fit R^2: PC2 = {rates.pc2_r2:.4f}, PC3 = {rates.pc3_r2:.4f}")

    # --- 12. Today's forward curve at T = 1, 5, 10 ---
    today_curve = rates.get_curve(day_index=-1)
    print("\nForward curve today:")
    for T in (1, 5, 10):
        print(f"  f(0, {T}) = {rates.get_rate(today_curve, T):.4%}")

    # --- 13. Simulated forward rates at tau = 0.08, 5, 25 ---
    simulated_paths = rates.simulate(n_steps=1000, n_paths=1, seed=42)
    simulated_curve = simulated_paths[0, -1, :]
    print("\nSimulated forward rates (one step, dt = 1/252):")
    for tau in (0.08, 5, 25):
        print(f"  f(0, {tau}) = {rates.get_rate(simulated_curve, tau):.4%}")

    # --- ZCB prices and yields, from today's forward curve ---
    T_grid, Z_today = rates.zcb()
    y_today = rates.compute_yield()
    print("\nZCB prices / yields today:")
    for T in (1, 5, 10):
        print(f"  Z(0, {T}) = {rates.get_rate(Z_today[1:], T):.6f}, "
              f"y(0, {T}) = {rates.get_rate(y_today, T):.4%}")

    # --- Sanity check: Z(0,T_i-1) * Z(0;T_i-1,T_i) = Z(0,T_i) ---
    rates.verify_forward_zcb()
    print("\nSanity check passed: Z(0,T_i-1) * Z(0; T_i-1,T_i) = Z(0,T_i) for all i")

    # --- Plots ---
    rates.plot_covariance_matrix()

    # Simulated forward curves at T = 0 (today), 1, 5, 10 years ahead,
    # under the Musiela HJM SDE, with the corresponding ZCB price and
    # yield curves.
    horizons = (0, 1, 5, 10)
    n_steps_curve = int(max(horizons) / rates.dt)
    curve_paths = rates.simulate(n_steps=n_steps_curve, n_paths=1, seed=7)
    horizon_curves = [curve_paths[0, int(round(h / rates.dt)), :] for h in horizons]
    horizon_labels = ["T = 0 (today)"] + [f"T = {h}" for h in horizons[1:]]

    horizon_zcbs = [rates.zcb(curve=c)[1] for c in horizon_curves]
    horizon_yields = [rates.compute_yield(curve=c) for c in horizon_curves]

    rates.plot_forward_curves(curves=horizon_curves, labels=horizon_labels)
    rates.plot_zcb_curves(curves=horizon_zcbs, labels=horizon_labels)
    rates.plot_yield_curves(curves=horizon_yields, labels=horizon_labels)

    rates.plot_simulated_rates(simulated_paths, tenors_to_track=(0.08, 5, 25))
