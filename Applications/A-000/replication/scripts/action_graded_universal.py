"""
action_graded_universal.py — Universal univariate estimator.

Generalises the action-graded estimator to ANY univariate linear process
of order ≤ 2 by parametrising the embedding plane in pure-time
coordinates:

    Δ² y_t = α y_t + β Δy_t + ε_t

Two parameters (α, β), zero regime assumptions. Transparently handles
stationary, non-stationary I(1), I(2), and explosive AR processes.

Conversion to classical AR(2) coefficients (φ_1, φ_2):
    φ_1 = (2 - β) / (1 - α - β)
    φ_2 = -1 / (1 - α - β)

(Derivation in the paper §6.)
"""
from __future__ import annotations

import numpy as np


def kinematic_embed(y: np.ndarray, d: int = 3) -> np.ndarray:
    y = np.asarray(y, dtype=float).ravel()
    T = len(y)
    cols = [y]
    for _ in range(1, d):
        cols.append(np.diff(cols[-1]))
    return np.column_stack([col[d - 1 - i: T - i] for i, col in enumerate(cols)])


def ab_to_phi(a: float, b: float) -> tuple[float, float]:
    """(a, b) → (φ_1, φ_2) for the equivalent AR(2).

    From the pure-time forward equation y_{t+1} = a y_t + b Δy_t + ε,
    using y_{t-1} = y_t - Δy_t:
        y_{t+1} = a y_t + b (y_t - y_{t-1}) = (a + b) y_t - b y_{t-1} + ε.
    Therefore φ_1 = a + b and φ_2 = -b.
    """
    return a + b, -b


def phi_to_ab(phi1: float, phi2: float) -> tuple[float, float]:
    """(φ_1, φ_2) → (a, b) inverse map. b = -φ_2, a = φ_1 + φ_2."""
    return phi1 + phi2, -phi2


# ---------------------------------------------------------------------
# Estimator: forward equation y_{t+1} = a y_t + b Δy_t + ε
# (level-1 closed form via OLS on (y_t, Δy_t) → y_{t+1})
# ---------------------------------------------------------------------

def ptls_universal_level1(y: np.ndarray) -> tuple[float, float]:
    """Level-1 universal estimator.

    Returns (â, b̂) by OLS regression of y_{t+1} on (y_t, Δy_t).
    No regime assumptions; valid for AR(1)/AR(2)/RW/I(2)/explosive.
    """
    y = np.asarray(y, dtype=float).ravel()
    if len(y) < 4:
        return 0.0, 0.0
    # y_t for t = 1..T-2 (so y_{t+1} valid + Δy_t = y_t - y_{t-1} valid)
    y_t = y[1:-1]
    dy_t = y[1:-1] - y[:-2]
    y_next = y[2:]
    X = np.column_stack([y_t, dy_t])
    coef, *_ = np.linalg.lstsq(X, y_next, rcond=None)
    return float(coef[0]), float(coef[1])


# ---------------------------------------------------------------------
# Estimator with graded action loss (level-1 + grades 0, 2, 3)
# ---------------------------------------------------------------------

def graded_residual_loss(a: float, b: float, y: np.ndarray,
                            taus: tuple[int, ...] = (1, 2, 3),
                            tri_pairs: tuple[tuple[int, int], ...] = ((1, 2),),
                            w_emb: float = 1.0, w0: float = 1.0,
                            w2: float = 1.0, w3: float = 1.0) -> float:
    """Action-graded loss on the residual ε_{t+1} = y_{t+1} - a y_t - b Δy_t.

    The residual is itself a univariate sequence; we embed it and apply
    the graded signature loss against the iid hypothesis (residual
    should be uncorrelated white noise across grades).
    """
    y = np.asarray(y, dtype=float).ravel()
    if len(y) < 5:
        return float("inf")
    y_t = y[1:-1]
    dy_t = y[1:-1] - y[:-2]
    y_next = y[2:]
    eps = y_next - a * y_t - b * dy_t

    # Level-1 (sum of squares of residual)
    L = w_emb * float(np.sum(eps * eps))

    if w0 == 0.0 and w2 == 0.0 and w3 == 0.0:
        return L

    # Build the embedding of the residual itself for higher-grade penalties.
    # ε is shorter than y by 2; we re-embed.
    if len(eps) < 4:
        return L
    Qr = kinematic_embed(eps, d=3)
    if Qr.shape[0] < max(taus) + 1:
        return L

    for tau in taus:
        if tau >= Qr.shape[0]:
            continue
        a, b = Qr[:-tau], Qr[tau:]
        if w0 != 0.0:
            sp = np.einsum("ti,ti->t", a, b)
            L += w0 * float(np.sum(sp * sp))
        if w2 != 0.0:
            w12 = a[:, 0] * b[:, 1] - a[:, 1] * b[:, 0]
            w13 = a[:, 0] * b[:, 2] - a[:, 2] * b[:, 0]
            w23 = a[:, 1] * b[:, 2] - a[:, 2] * b[:, 1]
            L += w2 * float(np.sum(w12 * w12 + w13 * w13 + w23 * w23))
    if w3 != 0.0:
        for (t1, t2) in tri_pairs:
            if t2 >= Qr.shape[0]:
                continue
            N = Qr.shape[0] - t2
            A = Qr[:N]
            B = Qr[t1: t1 + N]
            C = Qr[t2:]
            triv = (A[:, 0] * (B[:, 1] * C[:, 2] - B[:, 2] * C[:, 1])
                    - A[:, 1] * (B[:, 0] * C[:, 2] - B[:, 2] * C[:, 0])
                    + A[:, 2] * (B[:, 0] * C[:, 1] - B[:, 1] * C[:, 0]))
            L += w3 * float(np.sum(triv * triv))
    return L


def ptls_universal_graded(y: np.ndarray, w_emb=1.0, w0=1.0, w2=1.0, w3=1.0,
                              n_iters=30, tol=1e-12) -> tuple[float, float]:
    """Iterative refinement of (a, b) with full graded action loss.

    Starts from level-1 OLS estimate and refines via coordinate descent
    + golden section. Loss is non-quadratic in (a, b) due to the
    higher-grade terms.
    """
    a0, b0 = ptls_universal_level1(y)

    def loss_at(theta):
        return graded_residual_loss(theta[0], theta[1], y,
                                       w_emb=w_emb, w0=w0, w2=w2, w3=w3)

    def golden_1d(f, lo, hi, tol=1e-13, max_iter=120):
        phi = (np.sqrt(5.0) - 1.0) / 2.0
        c = hi - phi * (hi - lo)
        d = lo + phi * (hi - lo)
        fc, fd = f(c), f(d)
        for _ in range(max_iter):
            if abs(hi - lo) < tol:
                break
            if fc < fd:
                hi, d, fd = d, c, fc
                c = hi - phi * (hi - lo)
                fc = f(c)
            else:
                lo, c, fc = c, d, fd
                d = lo + phi * (hi - lo)
                fd = f(d)
        return 0.5 * (lo + hi)

    theta = [a0, b0]
    half = [max(0.5 * abs(a0), 0.5), max(0.5 * abs(b0), 0.5)]
    for k in range(n_iters):
        prev_loss = loss_at(theta)
        for i in range(2):
            best = golden_1d(
                lambda v, i=i: loss_at([theta[0] if i != 0 else v,
                                          theta[1] if i != 1 else v]),
                theta[i] - half[i], theta[i] + half[i],
            )
            theta[i] = best
        if abs(prev_loss - loss_at(theta)) < tol:
            break
        half = [h * 0.5 for h in half]
    return float(theta[0]), float(theta[1])


# ---------------------------------------------------------------------
# Baselines
# ---------------------------------------------------------------------

def ar2_ols(y: np.ndarray) -> tuple[float, float]:
    """Standard OLS estimate of AR(2): y_t = φ_1 y_{t-1} + φ_2 y_{t-2} + ε_t.

    Uses backward recursion in y. Most direct baseline.
    """
    y = np.asarray(y, dtype=float).ravel()
    if len(y) < 4:
        return 0.0, 0.0
    Y = y[2:]
    X = np.column_stack([y[1:-1], y[:-2]])
    coef, *_ = np.linalg.lstsq(X, Y, rcond=None)
    return float(coef[0]), float(coef[1])


def yule_walker_ar2(y: np.ndarray) -> tuple[float, float]:
    """Yule-Walker estimate of AR(2). Method-of-moments via autocorrelations.

    Assumes stationarity; biased on non-stationary processes.
    """
    y = np.asarray(y, dtype=float).ravel()
    y = y - y.mean()
    n = len(y)
    if n < 4:
        return 0.0, 0.0
    r0 = float((y * y).mean())
    r1 = float((y[:-1] * y[1:]).mean())
    r2 = float((y[:-2] * y[2:]).mean())
    if r0 < 1e-15:
        return 0.0, 0.0
    rho1 = r1 / r0
    rho2 = r2 / r0
    den = 1.0 - rho1 * rho1
    if abs(den) < 1e-15:
        return 0.0, 0.0
    phi1 = (rho1 - rho1 * rho2) / den
    phi2 = (rho2 - rho1 * rho1) / den
    return float(phi1), float(phi2)
