"""
action_graded_multivariate.py — Multivariate pure-time estimator.

Generalises PT-univ to a d-dimensional vector process:

    y_{t+1} = A y_t + B Δy_t + ε_{t+1}

with A, B ∈ R^{d×d}. Converts bijectively to VAR(2):
    Φ_1 = A + B,    Φ_2 = -B.

The matrix M := A + B - I is the natural ANALOGUE of the
error-correction matrix of VECM (it captures cointegration as a
rank-deficiency: rank(M) = r if there are r cointegrating relations).
Crucially, the estimator does NOT need to specify r in advance — it
emerges from the rank of the OLS estimate M̂.
"""
from __future__ import annotations

import numpy as np


def kinematic_embed_mv(Y: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """For a (T, d) panel, return (Y_t, ΔY_t, Y_next) aligned for the
    forward equation y_{t+1} = A y_t + B Δy_t + ε_{t+1}.

    Y_t      shape (T-2, d)  — y at time t (t = 1..T-2)
    ΔY_t     shape (T-2, d)  — Δy at time t = y_t - y_{t-1}
    Y_next   shape (T-2, d)  — y at time t+1
    """
    Y = np.asarray(Y, dtype=float)
    if Y.ndim == 1:
        Y = Y.reshape(-1, 1)
    T, d = Y.shape
    if T < 4:
        return np.zeros((0, d)), np.zeros((0, d)), np.zeros((0, d))
    Y_t = Y[1:-1]
    dY_t = Y[1:-1] - Y[:-2]
    Y_next = Y[2:]
    return Y_t, dY_t, Y_next


def ptmv_level1(Y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Multivariate pure-time level-1 estimator.

    OLS for (A, B) in y_{t+1} = A y_t + B Δy_t + ε.
    Stacks Y and ΔY as (T-2, 2d) and solves the multi-output linear
    regression Y_next = [A | B] [Y; ΔY]^T.
    """
    Y = np.asarray(Y, dtype=float)
    if Y.ndim == 1:
        Y = Y.reshape(-1, 1)
    Y_t, dY_t, Y_next = kinematic_embed_mv(Y)
    if Y_t.shape[0] == 0:
        d = Y.shape[1]
        return np.eye(d), np.zeros((d, d))
    X = np.hstack([Y_t, dY_t])              # (T-2, 2d)
    coef, *_ = np.linalg.lstsq(X, Y_next, rcond=None)  # (2d, d)
    d = Y.shape[1]
    A = coef[:d].T                           # transpose so A acts on column vector
    B = coef[d:].T
    return A, B


def AB_to_Phi(A: np.ndarray, B: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Convert pure-time (A, B) to classical VAR(2) (Φ_1, Φ_2).
        Φ_1 = A + B,    Φ_2 = -B.
    """
    return A + B, -B


def Phi_to_AB(Phi1: np.ndarray, Phi2: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Inverse: B = -Φ_2, A = Φ_1 + Φ_2."""
    return Phi1 + Phi2, -Phi2


def ec_matrix(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """Error-correction matrix M = A + B - I.

    rank(M) is the natural cointegration rank emerging from the
    PT-MV estimate (no a-priori specification needed).
    """
    d = A.shape[0]
    return A + B - np.eye(d)


def emergent_rank(M: np.ndarray, T: int, threshold: float = 2.0) -> tuple[int, np.ndarray]:
    """Johansen-style rank detection on M̂.

    A singular value σ_i is "effective" iff σ_i √T > threshold.
    Spurious (data-noise) eigenvalues have σ √T → bounded; real
    structure has σ √T → ∞.

    Returns (effective_rank, singular_values).
    """
    s = np.linalg.svd(M, compute_uv=False)
    # Cointegration rank = number of σ_i with σ √T > threshold
    sqrtT = np.sqrt(T)
    eff = int(np.sum(s * sqrtT > threshold))
    return eff, s


# ---------------------------------------------------------------------
# Baselines: VAR(2) OLS, VECM (Engle-Granger 2-step)
# ---------------------------------------------------------------------

def var2_ols(Y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Standard VAR(2) OLS: y_t = Φ_1 y_{t-1} + Φ_2 y_{t-2} + ε."""
    Y = np.asarray(Y, dtype=float)
    if Y.ndim == 1:
        Y = Y.reshape(-1, 1)
    T, d = Y.shape
    if T < 4:
        return np.eye(d), np.zeros((d, d))
    Y_target = Y[2:]
    X = np.hstack([Y[1:-1], Y[:-2]])
    coef, *_ = np.linalg.lstsq(X, Y_target, rcond=None)
    Phi1 = coef[:d].T
    Phi2 = coef[d:].T
    return Phi1, Phi2


def vecm_engle_granger_2step(Y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Engle-Granger 2-step VECM (assuming bivariate, rank-1 cointegration).

    Step 1: regress y_1 on y_2 → cointegrating vector β = (1, -β̂_21).
    Step 2: regress (Δy_1, Δy_2) on (lagged level residual, lagged Δy).

    Returns (Π, Γ_1) where Π = α β^T (rank 1 by construction in this
    bivariate variant) and Γ_1 is the short-run dynamics matrix.
    Bivariate-only baseline.
    """
    Y = np.asarray(Y, dtype=float)
    if Y.ndim != 2 or Y.shape[1] != 2:
        # Non-bivariate: skip
        d = Y.shape[1] if Y.ndim == 2 else 1
        return np.zeros((d, d)), np.zeros((d, d))
    T = Y.shape[0]
    if T < 5:
        return np.zeros((2, 2)), np.zeros((2, 2))

    y1 = Y[:, 0]; y2 = Y[:, 1]
    # Step 1 — regress y1 on y2 (with const)
    X1 = np.column_stack([np.ones(T), y2])
    coef1, *_ = np.linalg.lstsq(X1, y1, rcond=None)
    beta_const, beta_21 = coef1
    u = y1 - beta_const - beta_21 * y2          # residual = error correction term
    # Step 2 — Δy_t on (u_{t-1}, Δy_{t-1})
    dY = np.diff(Y, axis=0)              # (T-1, 2)
    u_lag = u[:-2]                       # (T-2,)
    dY_lag = dY[:-1]                     # (T-2, 2)
    target = dY[1:]                      # (T-2, 2)
    X2 = np.column_stack([u_lag, dY_lag])
    coef2, *_ = np.linalg.lstsq(X2, target, rcond=None)
    alpha = coef2[0]                     # (2,) — adjustment speeds
    Gamma1 = coef2[1:].T                 # (2, 2) short-run
    Pi = np.outer(alpha, np.array([1.0, -beta_21]))  # rank-1 EC matrix
    return Pi, Gamma1
