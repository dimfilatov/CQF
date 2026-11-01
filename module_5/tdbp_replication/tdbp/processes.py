"""Price models (GBM, Merton-Heston) and a simulator that adds discrete
dividends and daily price limits.

Simulated paths are a dict of arrays of shape (n_paths, n_columns):
"S" traded price, "Y" same shocks without dividends / price limits,
"V" variance (Merton-Heston), and the daily-price-limit state
("day_open", "absorbed", "sign", "stuck").
"""
import numpy as np


def normals(rng, n, antithetic):
    """Standard normals, optionally antithetic (z, -z)."""
    if not antithetic:
        return rng.standard_normal(n)
    z = rng.standard_normal((n + 1) // 2)
    return np.concatenate([z, -z])[:n]


def fold(x, lo, hi):
    """Mirror x into [lo, hi]; returns (folded x, number of reflections)."""
    w = hi - lo
    k = np.floor((x - lo) / w)
    rem = x - lo - k * w
    return np.where(k % 2 == 0, lo + rem, hi - rem), np.abs(k)


class GBM:
    """dS/S = r dt + sigma dW (exact log step)."""

    def __init__(self, r, sigma):
        self.r = r
        self.sigma = sigma

    def initial_state(self, S0, n):
        return {"S": np.full(n, float(S0))}

    def increment(self, state, dt, rng, antithetic):
        """Returns (d log S, updated extra state)."""
        z = normals(rng, len(state["S"]), antithetic)
        return (self.r - 0.5 * self.sigma ** 2) * dt + self.sigma * np.sqrt(dt) * z, {}


class MertonHeston:
    """Bates model (slide 59):
        dS/S = (r - lam E[J-1]) dt + sqrt(V) dW_S + (J-1) dN
        dV   = kappa (theta - V) dt + xi sqrt(V) dW_V,   d<W_S, W_V> = rho dt
        log J ~ N(mu_j - sigma_j^2/2, sigma_j^2)
    Full-truncation Euler for V; v0 is the initial variance."""

    def __init__(self, r, kappa, theta, xi, rho, v0, lam, mu_j, sigma_j):
        self.r = r
        self.kappa, self.theta, self.xi, self.rho, self.v0 = kappa, theta, xi, rho, v0
        self.lam, self.mu_j, self.sigma_j = lam, mu_j, sigma_j

    def initial_state(self, S0, n):
        return {"S": np.full(n, float(S0)), "V": np.full(n, self.v0)}

    def increment(self, state, dt, rng, antithetic):
        n, V = len(state["S"]), state["V"]
        Vp = np.maximum(V, 0.0)
        z1 = normals(rng, n, antithetic)
        z2 = self.rho * z1 + np.sqrt(1 - self.rho ** 2) * normals(rng, n, antithetic)
        n_jumps = rng.poisson(self.lam * dt, n)
        jumps = (n_jumps * (self.mu_j - 0.5 * self.sigma_j ** 2)
                 + np.sqrt(n_jumps) * self.sigma_j * rng.standard_normal(n))
        compensator = self.lam * (np.exp(self.mu_j) - 1.0)
        dlog = (self.r - compensator - 0.5 * Vp) * dt + np.sqrt(Vp * dt) * z1 + jumps
        V_new = V + self.kappa * (self.theta - Vp) * dt + self.xi * np.sqrt(Vp * dt) * z2
        return dlog, {"V": V_new}


class Simulator:
    """Simulates a model on steps 0..n_steps of size dt.

    dividends : list of (step, amount, kind); kind 'cash' (S -> S - amount) or
                'proportional' (S -> S (1 - amount)); S at ``step`` is ex-dividend.
    dpl_limit : daily price limit alpha = beta (None = no limits). Each trading
                day of ``steps_per_day`` steps the price stays in
                [(1 - alpha) S_open, (1 + alpha) S_open] (slide 72).
    dpl_boundary : 'absorbing'  - stays on the boundary for the rest of the day
                   'instant'    - mirrored at the boundary
                   'slow'       - sticks for a random time (mean ``dpl_rho`` steps),
                                  then the rest of the day's increments change sign
    """

    def __init__(self, model, n_steps, dt, dividends, dpl_limit, dpl_boundary, dpl_rho,
                 steps_per_day):
        self.model, self.n_steps, self.dt = model, n_steps, dt
        self.dividends = list(dividends)
        self.dpl_limit, self.dpl_boundary, self.dpl_rho = dpl_limit, dpl_boundary, dpl_rho
        self.steps_per_day = steps_per_day

    def initial_state(self, S0, n):
        state = self.model.initial_state(S0, n)
        state["Y"] = state["S"].copy()
        if self.dpl_limit is not None:
            self._new_day(state)
        return state

    def _new_day(self, state):
        n = len(state["S"])
        state["day_open"] = state["S"].copy()
        state["absorbed"] = np.zeros(n)        # 0 free, +1 at upper, -1 at lower boundary
        state["sign"] = np.ones(n)             # slow reflection: -1 after a hit
        state["stuck"] = np.zeros(n)           # slow reflection: steps left on the boundary

    def _apply_limit(self, x, dx, state, rng):
        """Constrained log-price; updates the boundary flags in ``state``."""
        lo = np.log(state["day_open"] * (1 - self.dpl_limit))
        hi = np.log(state["day_open"] * (1 + self.dpl_limit))
        if self.dpl_boundary == "absorbing":
            x_try = x + dx
            frozen = state["absorbed"] != 0
            hit = np.where(x_try >= hi, 1.0, np.where(x_try <= lo, -1.0, 0.0))
            state["absorbed"] = np.where(frozen, state["absorbed"], hit)
            return np.where(frozen, x, np.clip(x_try, lo, hi))
        if self.dpl_boundary == "instant":
            return fold(x + dx, lo, hi)[0]
        if self.dpl_boundary == "slow":
            sign, stuck = state["sign"], state["stuck"]
            x_try = x + sign * dx
            out = (x_try > hi) | (x_try < lo)
            if self.dpl_rho > 0:
                sojourn = rng.geometric(1.0 / (1.0 + self.dpl_rho), len(x)) - 1
                x_new = np.where(out, np.where(x_try > hi, hi, lo), x_try)
                new_stuck = np.where(out, sojourn, 0.0)
                new_sign = np.where(out, -sign, sign)
            else:
                x_new, k = fold(x_try, lo, hi)
                new_stuck = np.zeros_like(stuck)
                new_sign = sign * np.where(k % 2 == 1, -1.0, 1.0)
            is_stuck = stuck > 0
            state["sign"] = np.where(is_stuck, sign, new_sign)
            state["stuck"] = np.where(is_stuck, stuck - 1, new_stuck)
            return np.where(is_stuck, x, x_new)
        raise ValueError(f"unknown boundary {self.dpl_boundary!r}")

    def _advance(self, state, step, rng, antithetic):
        """Move the state from ``step`` to ``step + 1``."""
        dlog, extra = self.model.increment(state, self.dt, rng, antithetic)
        new = {**state, **extra}
        new["Y"] = state["Y"] * np.exp(dlog)
        if self.dpl_limit is None:
            new["S"] = state["S"] * np.exp(dlog)
        else:
            new["S"] = np.exp(self._apply_limit(np.log(state["S"]), dlog, new, rng))
        for d_step, amount, kind in self.dividends:
            if d_step == step + 1:
                new["S"] = np.maximum(new["S"] - amount, 1e-8) if kind == "cash" else new["S"] * (1 - amount)
        if self.dpl_limit is not None and (step + 1) % self.steps_per_day == 0:
            self._new_day(new)
        return new

    def simulate(self, n_paths, rng, S0=None, state0=None, start_step=0, antithetic=False):
        """Paths dict over steps start_step..n_steps. Start from ``S0`` or from
        ``state0`` (dict of scalars, used by nested Monte Carlo)."""
        if state0 is None:
            state = self.initial_state(S0, n_paths)
        else:
            state = {k: np.full(n_paths, float(v)) for k, v in state0.items()}
        m = self.n_steps - start_step
        paths = {k: np.empty((n_paths, m + 1)) for k in state}
        for k in state:
            paths[k][:, 0] = state[k]
        for j in range(m):
            state = self._advance(state, start_step + j, rng, antithetic)
            for k in paths:
                paths[k][:, j + 1] = state[k]
        return paths
