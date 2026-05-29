"""
Walkthrough — three-body, path (i): Koopman state augmentation.

Augmented state:
    z       = (r1, r2, r3, v1, v2, v3)                # 12-d native
    Ψ(z)    = (z, W12, W13, W23)                      # 15-d augmented

where W_{ij} = 1 / |r_j - r_i|^3 — the natural observables associated
with Newton's potential.

In Ψ-space:
    ṙ_i  = v_i                                  # linear in Ψ
    v̇_i  = G Σ_j m_j (r_j - r_i) W_{ij}         # bilinear in Ψ
    Ẇ_ij = -3 W_{ij}^{5/3} (r_j - r_i)·(v_j - v_i)
                                                 # not polynomial in Ψ alone

So bilinear closure does NOT obtain for the W components without further
augmentation. As anticipated from the pendulum smoke test (and as is
standard in EDMD), path (i) predicts Ψ(z_{t+1}) with O(dt²) Taylor-
truncation error per step; recovery converges to machine precision as
dt → 0.

The walkthrough's pedagogical point: path (ii) recovers the discrete map
exactly; path (i) recovers the Koopman operator in the continuous-time
limit. Two distinct readings, both legitimate.
"""
from __future__ import annotations

import json
import os
import time

import numpy as np


# ---------------------------------------------------------------------
# Native dynamics (same as path ii)
# ---------------------------------------------------------------------

def three_body_rhs(z, masses, G=1.0):
    r = z[:6].reshape(3, 2)
    v = z[6:].reshape(3, 2)
    a = np.zeros_like(r)
    for i in range(3):
        for j in range(3):
            if i == j: continue
            d = r[j] - r[i]
            r3 = (d @ d) ** 1.5
            a[i] += G * masses[j] * d / r3
    return np.concatenate([v.ravel(), a.ravel()])


def euler_integrate(z0, masses, G, dt, T):
    Z = np.empty((T, len(z0))); Z[0] = z0
    for t in range(1, T):
        Z[t] = Z[t-1] + dt * three_body_rhs(Z[t-1], masses, G)
    return Z


def figure_eight_ic():
    p1 = np.array([0.97000436, -0.24308753])
    p3 = -p1; p2 = np.zeros(2)
    v3 = np.array([-0.93240737, -0.86473146])
    v1 = -v3 / 2.0; v2 = -v3 / 2.0
    return np.concatenate([p1, p2, p3, v1, v2, v3])


# ---------------------------------------------------------------------
# Koopman augmentation
# ---------------------------------------------------------------------

def augment(Z):
    """z (T, 12) → Ψ(z) (T, 15) by appending W_{ij} = 1/r_ij^3 for each pair."""
    T = Z.shape[0]
    R = Z[:, :6].reshape(T, 3, 2)
    pairs = [(0, 1), (0, 2), (1, 2)]
    W = []
    for (i, j) in pairs:
        d = R[:, j] - R[:, i]
        r2 = np.sum(d * d, axis=1)
        W.append(1.0 / r2 ** 1.5)
    W = np.stack(W, axis=1)                         # (T, 3)
    return np.hstack([Z, W])                        # (T, 15)


def quad_design_mv(Psi):
    """Quadratic pure-time design — same structure as published code."""
    T, d = Psi.shape
    Psi_t = Psi[1:-1]
    dPsi_t = Psi[1:-1] - Psi[:-2]
    Psi_next = Psi[2:]
    feats = [np.ones((Psi_t.shape[0], 1)), Psi_t, dPsi_t,
             Psi_t * Psi_t, dPsi_t * dPsi_t]
    for j in range(d):
        for k in range(j + 1, d):
            feats.append((Psi_t[:, j] * Psi_t[:, k]).reshape(-1, 1))
    for j in range(d):
        for k in range(j + 1, d):
            feats.append((dPsi_t[:, j] * dPsi_t[:, k]).reshape(-1, 1))
    for j in range(d):
        for k in range(d):
            feats.append((Psi_t[:, j] * dPsi_t[:, k]).reshape(-1, 1))
    return np.hstack(feats), Psi_next


# ---------------------------------------------------------------------
# Fit + report
# ---------------------------------------------------------------------

def fit_evaluate(Z, label):
    Psi = augment(Z)
    X, Y = quad_design_mv(Psi)
    coef, *_ = np.linalg.lstsq(X, Y, rcond=None)
    pred = X @ coef
    resid = pred - Y
    rmse_per_psi = np.sqrt(np.mean(resid * resid, axis=0))
    rmse_total = float(np.sqrt(np.mean(np.sum(resid * resid, axis=1))))
    cond_X = float(np.linalg.cond(X))
    psi_names = (["r1x","r1y","r2x","r2y","r3x","r3y",
                  "v1x","v1y","v2x","v2y","v3x","v3y",
                  "W12","W13","W23"])
    print(f"\n=== {label} ===")
    print(f"  T_samples used:  {X.shape[0]}")
    print(f"  features:        {X.shape[1]}")
    print(f"  cond(X):         {cond_X:.3e}")
    print(f"  RMSE total Ψ:    {rmse_total:.4e}")
    print(f"  RMSE per-Ψ-comp:")
    for nm, r in zip(psi_names, rmse_per_psi):
        print(f"     {nm:>4s}: {r:.4e}")
    rmse_native = np.sqrt(np.mean(resid[:, :12] ** 2, axis=0))
    rmse_native_total = float(np.sqrt(np.mean(np.sum(resid[:, :12] ** 2, axis=1))))
    print(f"  RMSE on native state z (first 12):")
    print(f"     total: {rmse_native_total:.4e}")
    return {
        "label": label,
        "T_samples": int(X.shape[0]),
        "n_features": int(X.shape[1]),
        "cond_X": cond_X,
        "rmse_total_psi": rmse_total,
        "rmse_native_total": rmse_native_total,
        "rmse_per_psi_comp": {nm: float(r) for nm, r in zip(psi_names, rmse_per_psi)},
    }


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    results_dir = os.path.normpath(os.path.join(here, "..", "results"))
    os.makedirs(results_dir, exist_ok=True)
    out_path = os.path.join(results_dir, "walkthrough_three_body_path_i.json")

    print("=" * 72)
    print(" Walkthrough smoke test — Three-body path (i): Koopman augmentation")
    print("=" * 72)

    masses = np.array([1.0, 1.0, 1.0]); G = 1.0
    z0 = figure_eight_ic()

    configs = [
        ("dt=1e-2, T=5000",  1e-2, 5000),
        ("dt=1e-3, T=5000",  1e-3, 5000),
        ("dt=1e-4, T=5000",  1e-4, 5000),
        ("dt=1e-5, T=5000",  1e-5, 5000),
    ]

    all_results = {"runs": []}
    for label, dt, T in configs:
        t0 = time.time()
        Z = euler_integrate(z0, masses, G, dt, T)
        print(f"\n[Euler  {label}] integrated in {time.time()-t0:.2f}s")
        res = fit_evaluate(Z, f"Euler {label}")
        res["dt"] = dt; res["T"] = T
        res["integrator"] = "euler"
        all_results["runs"].append(res)

    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to: {out_path}")


if __name__ == "__main__":
    main()
