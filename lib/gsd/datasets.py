"""
gsd.datasets — small reproducible synthetic processes for demos and tests.

numpy only.
"""
from __future__ import annotations
import numpy as np

__all__ = ["rotation", "logistic", "cointegrated_var", "random_walk"]


def rotation(T=200, rho=0.97, omega=0.62, noise=0.02, seed=11):
    """Damped phase-space rotation observed through one coordinate.
    A clean oscillation a single real lag cannot reproduce."""
    rng = np.random.default_rng(seed)
    R = rho * np.array([[np.cos(omega), -np.sin(omega)],
                        [np.sin(omega), np.cos(omega)]])
    s = np.array([1.0, 0.0]); z = []
    for _ in range(T):
        z.append(s[0]); s = R @ s + noise * rng.standard_normal(2)
    return np.array(z)


def logistic(T=200, r=3.95, x0=0.4, burn=300):
    """Deterministic chaos: x_{t+1} = r x_t (1 - x_t). Near-zero linear
    autocorrelation; recovered only by a quadratic (grade-2) dictionary."""
    x = x0
    for _ in range(burn):
        x = r * x * (1 - x)
    out = []
    for _ in range(T):
        x = r * x * (1 - x); out.append(x)
    return np.array(out)


def random_walk(T=200, drift=0.0, sigma=1.0, seed=0):
    """Unit-root (integrated) process: spectral radius ~ 1."""
    rng = np.random.default_rng(seed)
    return np.cumsum(drift + sigma * rng.standard_normal(T))


def cointegrated_var(T=400, beta=(1.0, -1.0), alpha=-0.15, sigma=0.4, seed=3):
    """Two I(1) series sharing one stationary (cointegrating) combination
    b·y = beta0 y0 + beta1 y1. Built from a common stochastic trend plus a
    mean-reverting spread, the canonical case where a VAR in levels is
    mis-specified and the levels are individually non-stationary."""
    rng = np.random.default_rng(seed)
    trend = np.cumsum(sigma * rng.standard_normal(T))     # common I(1) trend
    spread = np.zeros(T)                                   # stationary AR(1)
    for t in range(1, T):
        spread[t] = (1 + alpha) * spread[t - 1] + 0.3 * rng.standard_normal()
    b0, b1 = beta
    y0 = trend + spread / b0
    y1 = (b0 * y0 - spread) / (-b1)                        # so b0 y0 + b1 y1 = spread (stationary)
    return np.column_stack([y0, y1])
