"""Reference prices (Black-Scholes, barrier closed form, Merton-Heston Fourier)
and the nested Monte Carlo engine (slide 75)."""
import numpy as np
from scipy.stats import norm

N = norm.cdf
trapezoid = getattr(np, "trapezoid", None) or np.trapz


class BlackScholes:
    def __init__(self, r, sigma):
        self.r, self.sigma = r, sigma

    def price(self, kind, S, K, tau):
        """kind: forward | call | put | digital_bond | digital_share."""
        S = np.asarray(S, dtype=float)
        r, s = self.r, self.sigma
        if kind == "forward":
            return S - K * np.exp(-r * tau)
        if tau <= 0:
            return {"call": np.maximum(S - K, 0.0), "put": np.maximum(K - S, 0.0),
                    "digital_bond": (S > K) * 1.0, "digital_share": S * (S > K)}[kind]
        d1 = (np.log(S / K) + (r + 0.5 * s * s) * tau) / (s * np.sqrt(tau))
        d2 = d1 - s * np.sqrt(tau)
        df = np.exp(-r * tau)
        return {"call": S * N(d1) - K * df * N(d2),
                "put": K * df * N(-d2) - S * N(-d1),
                "digital_bond": df * N(d2),
                "digital_share": S * N(d1)}[kind]

    def knock_out(self, kind, S, K, H, tau):
        """Reiner-Rubinstein, no rebate. kind: 'DOP' (down-and-out put, K > H)
        or 'UOC' (up-and-out call, K < H). Re-prices from the current spot."""
        S = np.asarray(S, dtype=float)
        down = kind == "DOP"
        inside = S > H if down else S < H
        if tau <= 0:
            return np.maximum(K - S, 0.0) * inside if down else np.maximum(S - K, 0.0) * inside
        if (down and K <= H) or (not down and K >= H):
            return np.zeros_like(S)
        phi, eta = (-1.0, 1.0) if down else (1.0, -1.0)
        r, s = self.r, self.sigma
        S = np.maximum(S, 1e-12)
        sT = s * np.sqrt(tau)
        mu = (r - 0.5 * s * s) / (s * s)
        x1 = np.log(S / K) / sT + (1 + mu) * sT
        x2 = np.log(S / H) / sT + (1 + mu) * sT
        y1 = np.log(H * H / (S * K)) / sT + (1 + mu) * sT
        y2 = np.log(H / S) / sT + (1 + mu) * sT
        df = np.exp(-r * tau)
        h1, h0 = (H / S) ** (2 * (mu + 1)), (H / S) ** (2 * mu)
        A = phi * S * N(phi * x1) - phi * K * df * N(phi * x1 - phi * sT)
        B = phi * S * N(phi * x2) - phi * K * df * N(phi * x2 - phi * sT)
        C = phi * S * h1 * N(eta * y1) - phi * K * df * h0 * N(eta * y1 - eta * sT)
        D = phi * S * h1 * N(eta * y2) - phi * K * df * h0 * N(eta * y2 - eta * sT)
        return np.where(inside, A - B + C - D, 0.0)


class BatesFourier:
    """Merton-Heston (Bates) price by Gil-Pelaez inversion of the characteristic
    function ("little trap" Heston form times the Merton jump term)."""

    def __init__(self, model, u_max, n_u):
        self.m = model
        self.u = np.linspace(1e-6, u_max, n_u)

    def _cf(self, w, logS, V, tau):
        """E[exp(i w log S_T)]; w complex (1, n_u), logS and V of shape (n, 1)."""
        m = self.m
        iw = 1j * w
        b = m.kappa - m.rho * m.xi * iw
        d = np.sqrt(b * b + m.xi ** 2 * (iw + w * w))
        g = (b - d) / (b + d)
        e = np.exp(-d * tau)
        C = ((m.r - m.lam * (np.exp(m.mu_j) - 1)) * iw * tau
             + m.kappa * m.theta / m.xi ** 2 * ((b - d) * tau - 2 * np.log((1 - g * e) / (1 - g))))
        D = (b - d) / m.xi ** 2 * (1 - e) / (1 - g * e)
        J = m.lam * tau * (np.exp(iw * (m.mu_j - 0.5 * m.sigma_j ** 2) - 0.5 * m.sigma_j ** 2 * w * w) - 1)
        return np.exp(C + D * V + J + iw * logS)

    def price(self, kind, S, V, K, tau, chunk=500):
        """kind: call | put; S, V arrays of the same length."""
        S, V = np.asarray(S, float), np.maximum(np.asarray(V, float), 0.0)
        r = self.m.r
        if tau <= 0:
            call = np.maximum(S - K, 0.0)
        else:
            call = np.empty_like(S)
            u = self.u[None, :]
            kern = np.exp(-1j * u * np.log(K)) / (1j * u)
            for s in range(0, len(S), chunk):
                logS, v = np.log(S[s:s + chunk])[:, None], V[s:s + chunk][:, None]
                f2 = self._cf(u + 0j, logS, v, tau)
                f1 = self._cf(u - 1j, logS, v, tau) / np.exp(logS + r * tau)
                P1 = 0.5 + trapezoid(np.real(kern * f1), self.u, axis=1) / np.pi
                P2 = 0.5 + trapezoid(np.real(kern * f2), self.u, axis=1) / np.pi
                call[s:s + chunk] = S[s:s + chunk] * P1 - K * np.exp(-r * tau) * P2
            call = np.maximum(call, 0.0)
        return call if kind == "call" else call - S + K * np.exp(-r * tau)


class MonteCarlo:
    """Nested Monte Carlo: from a state at ``step`` simulate fresh paths to
    maturity and average the discounted payoff. Barrier claims restart alive
    (re-pricing, as on slides 30-43)."""

    def __init__(self, simulator, payoff, n_paths, chunk, antithetic, seed):
        self.sim, self.payoff = simulator, payoff
        self.n_paths, self.chunk, self.antithetic = n_paths, chunk, antithetic
        self.rng = np.random.default_rng(seed)
        self.keys = list(simulator.initial_state(1.0, 1))      # state variables to restart from

    def price_state(self, state, step):
        total, done = 0.0, 0
        while done < self.n_paths:
            n = min(self.chunk, self.n_paths - done)
            paths = self.sim.simulate(n, self.rng, state0=state, start_step=step,
                                      antithetic=self.antithetic)
            total += self.payoff.path_value(paths["S"]).sum()
            done += n
        tau = (self.sim.n_steps - step) * self.sim.dt
        return np.exp(-self.sim.model.r * tau) * total / done

    def price_paths(self, paths, step, n_points):
        """Prices from the states of the first ``n_points`` paths at ``step``
        (identical states, e.g. all paths at t = 0, are priced once)."""
        n = min(n_points, len(paths["S"]))
        states = np.column_stack([paths[k][:n, step] for k in self.keys])
        unique, inverse = np.unique(states, axis=0, return_inverse=True)
        prices = np.array([self.price_state(dict(zip(self.keys, row)), step) for row in unique])
        return prices[np.asarray(inverse).ravel()]
