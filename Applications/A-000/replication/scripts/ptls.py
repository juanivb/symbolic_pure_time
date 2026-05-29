"""
ptls.py — PTLS / MLS estimators, paper-private prototype.

This module is the local working prototype of the Pure-Time Least Squares
estimator family for Paper 30. Once the smoke tests stabilise, the
canonical primitives graduate to ``gsd.estimation`` and this file is
retired.

Conventions
-----------
* All embeddings use raw (no-whitening) kinematic coordinates,
  ``q(t) = (u, Δu, Δ²u)``, so β has its native scale.
* Pesos Plancherel canónicos: w_g = 2j(g)+1 mapped to grades 0,1,2,3.
  We use the "graded products" convention: grade 0 (scalar product),
  grade 2 (bivector wedge), grade 3 (trivector / pseudoscalar).
* All estimators here implement formulation A (β scalar). MLS
  (formulation B, β multivector) lives in ``mls.py``.
"""
from __future__ import annotations

import numpy as np


# ---------------------------------------------------------------------
# Kinematic embedding (raw, no whitening)
# ---------------------------------------------------------------------

def kinematic_embed_raw(u: np.ndarray, d: int = 3) -> np.ndarray:
    """Return shape ``(T - (d-1), d)`` raw embedding of u: cols [u, Δu, Δ²u, ...].

    No standardisation. Used so that β retains its raw-scale meaning
    in regression contexts.
    """
    u = np.asarray(u, dtype=float).ravel()
    T = len(u)
    cols = [u]
    for _ in range(1, d):
        cols.append(np.diff(cols[-1]))
    # Align all columns to the shortest one.
    Q = np.column_stack([col[d - 1 - i: T - i] for i, col in enumerate(cols)])
    return Q


# ---------------------------------------------------------------------
# Standard OLS (scalar regression, single regressor)
# ---------------------------------------------------------------------

def ols(y: np.ndarray, x: np.ndarray) -> float:
    """OLS slope of y on x without intercept. Single-regressor case."""
    x = np.asarray(x, dtype=float).ravel()
    y = np.asarray(y, dtype=float).ravel()
    return float((x @ y) / (x @ x))


# ---------------------------------------------------------------------
# PTLS Level 1 — pointwise embedded squared norm (closed form)
# ---------------------------------------------------------------------

def ptls_l1(qy: np.ndarray, qx: np.ndarray) -> float:
    """
    PTLS Level 1.

    Minimises ``Σ_t ||q_y(t) − β q_x(t)||²`` over scalar β.
    Equivalent to GLS-style stacking of (level, velocity, acceleration)
    regressions with identity weights. Closed form:

        β̂_L1 = Σ_t ⟨q_x(t), q_y(t)⟩ / Σ_t ||q_x(t)||²
    """
    num = float(np.sum(qx * qy))
    den = float(np.sum(qx * qx))
    return num / den


# ---------------------------------------------------------------------
# PTLS Level 2 — graded loss with lagged scalar + bivector
# ---------------------------------------------------------------------

def _grade02_lagged_loss(r: np.ndarray, taus: tuple[int, ...],
                            w_scalar: float, w_bivec: float) -> float:
    """Sum of squares of lagged scalar product + lagged bivector wedge."""
    L = 0.0
    for tau in taus:
        r0 = r[:-tau]
        r1 = r[tau:]
        if w_scalar != 0.0:
            sp = np.einsum("ti,ti->t", r0, r1)
            L += w_scalar * float(np.sum(sp * sp))
        if w_bivec != 0.0:
            w12 = r0[:, 0] * r1[:, 1] - r0[:, 1] * r1[:, 0]
            w13 = r0[:, 0] * r1[:, 2] - r0[:, 2] * r1[:, 0]
            w23 = r0[:, 1] * r1[:, 2] - r0[:, 2] * r1[:, 1]
            L += w_bivec * float(np.sum(w12 * w12 + w13 * w13 + w23 * w23))
    return L


def _trivector_lagged_loss(r: np.ndarray, tau1: int, tau2: int,
                              w_triv: float) -> float:
    """Sum of squares of trivector det(r(t), r(t+τ₁), r(t+τ₂))."""
    if w_triv == 0.0:
        return 0.0
    N = r.shape[0]
    if N <= tau2:
        return 0.0
    a = r[: N - tau2]
    b = r[tau1: N - tau2 + tau1]
    c = r[tau2:]
    triv = (a[:, 0] * (b[:, 1] * c[:, 2] - b[:, 2] * c[:, 1])
            - a[:, 1] * (b[:, 0] * c[:, 2] - b[:, 2] * c[:, 0])
            + a[:, 2] * (b[:, 0] * c[:, 1] - b[:, 1] * c[:, 0]))
    return w_triv * float(np.sum(triv * triv))


def ptls_l2_loss(beta: float, qy: np.ndarray, qx: np.ndarray,
                    taus: tuple[int, ...] = (1, 2, 3),
                    w0: float = 1.0, w2: float = 1.0, w3: float = 0.0,
                    triv_taus: tuple[int, int] = (1, 2)) -> float:
    """
    Full PTLS-L2 loss. Pesos por grado:
      w0 — pointwise embedded norm + lagged scalar product (grade 0).
      w2 — lagged bivector wedge (grade 2).
      w3 — lagged trivector / pseudoscalar (grade 3).

    Default w3 = 0 (level-2 loss only). Set w3 > 0 to engage the
    pseudoscalar channel — that's effectively the Level-3 estimator.
    """
    r = qy - beta * qx
    L = w0 * float(np.sum(r * r))                            # ||r(t)||²
    L += _grade02_lagged_loss(r, taus, w_scalar=w0, w_bivec=w2)
    if w3 != 0.0:
        L += _trivector_lagged_loss(r, triv_taus[0], triv_taus[1], w3)
    return L


def ptls_l2(qy: np.ndarray, qx: np.ndarray,
                taus: tuple[int, ...] = (1, 2, 3),
                w0: float = 1.0, w2: float = 1.0, w3: float = 0.0,
                triv_taus: tuple[int, int] = (1, 2),
                bracket: tuple[float, float] = (-5.0, 5.0),
                tol: float = 1e-9, max_iter: int = 100) -> float:
    """
    PTLS Level 2 (or 3 with w3 > 0) via golden-section search.

    Loss: pointwise embedded squared norm + lagged graded products.
    The 1-D minimisation is robust because the loss is unimodal in β
    near the OLS optimum (verified empirically on calibration DGPs).
    """
    phi = (np.sqrt(5.0) - 1.0) / 2.0  # golden ratio inverse, ≈ 0.618
    a, b = bracket
    c = b - phi * (b - a)
    d = a + phi * (b - a)
    fc = ptls_l2_loss(c, qy, qx, taus, w0, w2, w3, triv_taus)
    fd = ptls_l2_loss(d, qy, qx, taus, w0, w2, w3, triv_taus)
    for _ in range(max_iter):
        if abs(b - a) < tol:
            break
        if fc < fd:
            b = d
            d = c
            fd = fc
            c = b - phi * (b - a)
            fc = ptls_l2_loss(c, qy, qx, taus, w0, w2, w3, triv_taus)
        else:
            a = c
            c = d
            fc = fd
            d = a + phi * (b - a)
            fd = ptls_l2_loss(d, qy, qx, taus, w0, w2, w3, triv_taus)
    return float(0.5 * (a + b))
