"""Payoffs (slides 14-17, 27): forward, call, put, digitals, knock-out barriers."""
import numpy as np


class Payoff:
    """kind: forward | call | put | digital_bond | digital_share.

    smoothing : width of a logistic smoothing of the digital step (None = sharp).
    barrier   : knock-out level (None = no barrier); direction 'up' or 'down'.
    bridge    : (sigma, dt) for the Brownian-bridge survival probability p_i of
                slide 15; None = discrete monitoring on the grid points.

    S arguments are path arrays of shape (n_paths, n_columns).
    """

    def __init__(self, kind, K, smoothing, barrier, direction, bridge):
        self.kind, self.K, self.smoothing = kind, K, smoothing
        self.barrier, self.direction, self.bridge = barrier, direction, bridge

    def terminal(self, S_T):
        K = self.K
        if self.kind == "forward":
            return S_T - K
        if self.kind == "call":
            return np.maximum(S_T - K, 0.0)
        if self.kind == "put":
            return np.maximum(K - S_T, 0.0)
        if self.smoothing:
            step = 0.5 * (1.0 + np.tanh(0.5 * (S_T - K) / self.smoothing))
        else:
            step = (S_T > K).astype(float)
        if self.kind == "digital_bond":
            return step
        if self.kind == "digital_share":
            return S_T * step
        raise ValueError(f"unknown payoff {self.kind!r}")

    def inside(self, S):
        """True where the price is on the alive side of the barrier."""
        if self.barrier is None:
            return np.ones_like(S, dtype=bool)
        return S > self.barrier if self.direction == "down" else S < self.barrier

    def alive(self, S):
        """alive[:, c]: barrier not touched on columns 0..c."""
        return np.logical_and.accumulate(self.inside(S), axis=1)

    def survival(self, S):
        """survival[:, c]: probability of staying alive from column c to c + 1."""
        alive = self.alive(S)
        surv = alive[:, 1:].astype(float)
        if self.barrier is not None and self.bridge is not None:
            sigma, dt = self.bridge
            ok = alive[:, 1:]
            l0 = np.log(np.where(ok, S[:, :-1], 2 * self.barrier) / self.barrier)
            l1 = np.log(np.where(ok, S[:, 1:], 2 * self.barrier) / self.barrier)
            surv *= np.clip(1.0 - np.exp(-2.0 * l0 * l1 / (sigma ** 2 * dt)), 0.0, 1.0)
        return surv

    def path_value(self, S):
        """Undiscounted g(S(1:n)) = g(S_n) * prod_i p_i."""
        return self.terminal(S[:, -1]) * np.prod(self.survival(S), axis=1)
