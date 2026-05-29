"""
action_graded_quadratic.py — Quadratic extension of the universal
pure-time estimator.

Linear universal:
    y_{t+1} = a y_t + b Δy_t + ε

Quadratic universal:
    y_{t+1} = c_0 + c_1 y_t + c_2 Δy_t + c_3 y_t² + c_4 (Δy_t)² + c_5 y_t·Δy_t + ε

Native parametrisations:
- Hénon  y_{t+1} = 1 - a y_t² + b y_{t-1} = 1 + b y_t - b Δy_t - a y_t²
  → (c_0, c_1, c_2, c_3, c_4, c_5) = (1, b, -b, -a, 0, 0).
- Logistic y_{t+1} = r y_t (1 - y_t) = r y_t - r y_t²
  → (c_0, c_1, c_2, c_3, c_4, c_5) = (0, r, 0, -r, 0, 0).

OLS in this 6-coordinate basis recovers chaotic parameters exactly
on noise-free trajectories, and at order 1/√T under noise.
"""
from __future__ import annotations

import numpy as np


def design_quadratic(y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return (X, y_next) for the quadratic pure-time regression."""
    y = np.asarray(y, dtype=float).ravel()
    if len(y) < 4:
        return np.zeros((0, 6)), np.zeros(0)
    y_t = y[1:-1]
    dy_t = y[1:-1] - y[:-2]
    y_next = y[2:]
    X = np.column_stack([
        np.ones_like(y_t),
        y_t,
        dy_t,
        y_t * y_t,
        dy_t * dy_t,
        y_t * dy_t,
    ])
    return X, y_next


def ptls_quadratic(y: np.ndarray) -> np.ndarray:
    """OLS in pure-time quadratic basis. Returns 6-vector of coefficients
    (c_0, c_1, c_2, c_3, c_4, c_5)."""
    X, y_next = design_quadratic(y)
    if X.shape[0] == 0:
        return np.zeros(6)
    coef, *_ = np.linalg.lstsq(X, y_next, rcond=None)
    return coef


def henon_params_from_quadratic(c: np.ndarray) -> tuple[float, float]:
    """Recover Hénon (a, b) from quadratic coefficients.
    Hénon: c_0=1, c_1=b, c_2=-b, c_3=-a, c_4=0, c_5=0.
    Returns (a, b)."""
    return -c[3], c[1]


def logistic_param_from_quadratic(c: np.ndarray) -> float:
    """Recover logistic r from quadratic coefficients.
    Logistic: c_0=0, c_1=r, c_2=0, c_3=-r, c_4=0, c_5=0."""
    # Use both c_1 and -c_3 as estimates and average.
    return 0.5 * (c[1] + (-c[3]))
