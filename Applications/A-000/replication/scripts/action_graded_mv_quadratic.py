"""
action_graded_mv_quadratic.py — Multivariate quadratic extension.

Extends the pure-time forward equation to multivariate bounded
nonlinear systems:

    y_{i, t+1} = c_{i,0} + Σ_j A_{ij} y_{j,t} + Σ_j B_{ij} Δy_{j,t}
                + Σ_j C_{ij} y_{j,t}² + Σ_j D_{ij} (Δy_{j,t})²
                + Σ_{j<k} E_{i,jk} y_{j,t} y_{k,t}    (cross y-y)
                + Σ_{j<k} F_{i,jk} Δy_{j,t} Δy_{k,t}  (cross Δ-Δ)
                + Σ_{j,k} G_{i,jk} y_{j,t} Δy_{k,t}   (cross y-Δ)
                + ε_{i, t+1}

Native parametrisations covered:
- Lorenz system (Euler-discretised): dx/dt = σ(y-x), dy/dt = x(ρ-z)-y,
  dz/dt = xy - βz. Linear in (x,y,z) plus cross terms xy, xz.
- Coupled logistic maps with synchronisation coupling.
- Hénon 2-D: y_{t+1} = 1 - a y_t² + b y_{t-1}, here as a bivariate
  state-space y = (y_t, y_{t-1}).
"""
from __future__ import annotations

from typing import Tuple

import numpy as np


def _quad_design_mv(Y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Build (X, Y_next) for the multivariate quadratic regression.

    For Y of shape (T, d), returns:
      X        — (T-2, n_features) design matrix
      Y_next   — (T-2, d) target

    The feature set per row at time t is:
      1, y_{t,1..d}, Δy_{t,1..d}, y_{t,j}² (j=1..d), Δy_{t,j}² (j=1..d),
      cross y_{t,j} y_{t,k} (j<k), cross Δy_{t,j} Δy_{t,k} (j<k),
      cross y_{t,j} Δy_{t,k} (all j, k).

    Total features: 1 + 2d + 2d + 2·d(d-1)/2 + d² = 1 + 4d + d(d-1) + d²
                  = 1 + 4d + 2d² - d = 1 + 3d + 2d².
    """
    Y = np.asarray(Y, dtype=float)
    if Y.ndim == 1:
        Y = Y.reshape(-1, 1)
    T, d = Y.shape
    if T < 4:
        return np.zeros((0, 1 + 3 * d + 2 * d * d)), np.zeros((0, d))
    Y_t = Y[1:-1]
    dY_t = Y[1:-1] - Y[:-2]
    Y_next = Y[2:]
    N = Y_t.shape[0]
    feats = [np.ones((N, 1))]                     # intercept
    feats.append(Y_t)                              # linear y
    feats.append(dY_t)                             # linear Δy
    feats.append(Y_t * Y_t)                        # diag squares y_j²
    feats.append(dY_t * dY_t)                      # diag squares Δy_j²
    # cross y y_{j} y_{k} (j<k)
    for j in range(d):
        for k in range(j + 1, d):
            feats.append((Y_t[:, j] * Y_t[:, k]).reshape(-1, 1))
    # cross Δ Δ
    for j in range(d):
        for k in range(j + 1, d):
            feats.append((dY_t[:, j] * dY_t[:, k]).reshape(-1, 1))
    # cross y Δy (all pairs)
    for j in range(d):
        for k in range(d):
            feats.append((Y_t[:, j] * dY_t[:, k]).reshape(-1, 1))
    X = np.hstack(feats)
    return X, Y_next


def feature_names(d: int) -> list[str]:
    """Human-readable feature names for the multivariate quadratic basis."""
    names = ["1"]
    names += [f"y_{j+1}" for j in range(d)]
    names += [f"dy_{j+1}" for j in range(d)]
    names += [f"y_{j+1}^2" for j in range(d)]
    names += [f"dy_{j+1}^2" for j in range(d)]
    for j in range(d):
        for k in range(j + 1, d):
            names.append(f"y_{j+1}*y_{k+1}")
    for j in range(d):
        for k in range(j + 1, d):
            names.append(f"dy_{j+1}*dy_{k+1}")
    for j in range(d):
        for k in range(d):
            names.append(f"y_{j+1}*dy_{k+1}")
    return names


def ptmv_quadratic(Y: np.ndarray) -> np.ndarray:
    """OLS in multivariate pure-time quadratic basis.

    Returns coefficient matrix of shape ``(n_features, d)``: row $f$
    column $i$ is the coefficient of feature $f$ in the equation for
    $y_{i, t+1}$.
    """
    X, Y_next = _quad_design_mv(Y)
    if X.shape[0] == 0:
        d = Y.shape[1] if Y.ndim == 2 else 1
        return np.zeros((1 + 3 * d + 2 * d * d, d))
    coef, *_ = np.linalg.lstsq(X, Y_next, rcond=None)
    return coef


# ---------------------------------------------------------------------
# Multivariate chaotic DGPs
# ---------------------------------------------------------------------

def lorenz_euler(T: int, dt: float = 0.01,
                    sigma: float = 10.0, rho: float = 28.0, beta: float = 8.0 / 3.0,
                    init: tuple = (1.0, 1.0, 1.0)) -> np.ndarray:
    """Lorenz '63 system, Euler-discretised. Returns (T, 3) trajectory."""
    Y = np.empty((T, 3))
    Y[0] = init
    for t in range(1, T):
        x, y, z = Y[t - 1]
        Y[t, 0] = x + dt * sigma * (y - x)
        Y[t, 1] = y + dt * (x * (rho - z) - y)
        Y[t, 2] = z + dt * (x * y - beta * z)
    return Y


def coupled_logistic(T: int, r: float = 3.95, eps_couple: float = 0.05,
                       init: tuple = (0.4, 0.6)) -> np.ndarray:
    """Two coupled logistic maps with diffusive coupling
    y_{i, t+1} = (1 - eps) f_r(y_{i, t}) + eps f_r(y_{j, t})."""
    Y = np.empty((T, 2))
    Y[0] = init
    for t in range(1, T):
        y1, y2 = Y[t - 1]
        f1 = r * y1 * (1 - y1)
        f2 = r * y2 * (1 - y2)
        Y[t, 0] = (1 - eps_couple) * f1 + eps_couple * f2
        Y[t, 1] = (1 - eps_couple) * f2 + eps_couple * f1
    return Y


def rossler_euler(T: int, dt: float = 0.05,
                     a: float = 0.2, b: float = 0.2, c: float = 5.7,
                     init: tuple = (0.1, 0.1, 0.1)) -> np.ndarray:
    """Rössler system, Euler-discretised."""
    Y = np.empty((T, 3))
    Y[0] = init
    for t in range(1, T):
        x, y, z = Y[t - 1]
        Y[t, 0] = x + dt * (-y - z)
        Y[t, 1] = y + dt * (x + a * y)
        Y[t, 2] = z + dt * (b + z * (x - c))
    return Y
