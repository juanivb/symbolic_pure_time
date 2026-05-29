"""
Walkthrough — double pendulum, path (i): Koopman state augmentation.

Path (i) thesis: instead of enriching FEATURES with system-aware nonlinear
terms (path ii), augment the STATE itself with observables that linearise
the dynamics. The augmentation is principled by the geometry of the
system (angles live on a circle → their natural observables are
(sin, cos)), not by reading the EOM term-by-term.

For the double pendulum:

    z       = (θ1, θ2, ω1, ω2)                        # 4-d native state
    Ψ(z)    = (θ1, θ2, ω1, ω2,
               s1, c1, s2, c2,
               ξ)                                     # 9-d augmented state

where  s_i = sin θ_i,  c_i = cos θ_i,
       ξ   = 1 / D,    D = 3 - cos(2(θ1 - θ2))  =  2 + 2 sin²(θ1 - θ2).

The dynamics in Ψ-space is *polynomial of degree ≤ 2* in Ψ:

    ṡ_i = c_i ω_i                               # bilinear
    ċ_i = -s_i ω_i                              # bilinear
    ω̇_i = ξ · N_i(Ψ)                            # bilinear in Ψ × ξ
    ξ̇   = -ξ² · Ḋ                               # polynomial in Ψ, ξ

So PTLS with the standard QUADRATIC basis on the augmented state Ψ — the
same `quad_design_mv` machinery used for Lorenz '63 in the published
paper — should recover the dynamics to machine precision under Euler
discretisation. This is the Koopman / EDMD reading of PTLS.

Comparison with path (ii) — same script structure, but the diccionary is
the auto-generated quadratic on Ψ rather than a hand-picked atomic-term
library.
"""
from __future__ import annotations

import json
import os
import time

import numpy as np


G = 9.81


# ---------------------------------------------------------------------
# Native pendulum dynamics (same as path ii)
# ---------------------------------------------------------------------

def pendulum_rhs(z, g=G):
    th1, th2, w1, w2 = z
    phi = th1 - th2
    s_phi = np.sin(phi); c_phi = np.cos(phi)
    D = 3.0 - np.cos(2 * phi)
    num1 = (-3 * g * np.sin(th1)
            - g * np.sin(th1 - 2 * th2)
            - 2 * s_phi * (w2 * w2 + w1 * w1 * c_phi))
    num2 = (2 * s_phi * (2 * w1 * w1 + 2 * g * np.cos(th1)
                          + w2 * w2 * c_phi))
    return np.array([w1, w2, num1 / D, num2 / D])


def euler_integrate(z0, dt, T):
    Z = np.empty((T, 4)); Z[0] = z0
    for t in range(1, T):
        Z[t] = Z[t-1] + dt * pendulum_rhs(Z[t-1])
    return Z


# ---------------------------------------------------------------------
# Koopman augmentation
# ---------------------------------------------------------------------

def augment(Z):
    """z = (θ1, θ2, ω1, ω2)  →  Ψ(z) = (z, sin θ1, cos θ1, sin θ2, cos θ2, ξ).

    Returns Psi shape (T, 9).
    """
    th1 = Z[:, 0]; th2 = Z[:, 1]; w1 = Z[:, 2]; w2 = Z[:, 3]
    phi = th1 - th2
    D = 3.0 - np.cos(2 * phi)
    Psi = np.column_stack([
        th1, th2, w1, w2,
        np.sin(th1), np.cos(th1),
        np.sin(th2), np.cos(th2),
        1.0 / D,
    ])
    return Psi


def quad_design_mv(Psi):
    """Quadratic pure-time design on a multivariate state — identical
    structure to the published `quad_design_mv` used for Lorenz."""
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
    psi_names = ["θ1","θ2","ω1","ω2", "s1","c1","s2","c2", "ξ"]
    print(f"\n=== {label} ===")
    print(f"  T_samples used:  {X.shape[0]}")
    print(f"  features:        {X.shape[1]}")
    print(f"  cond(X):         {cond_X:.3e}")
    print(f"  RMSE total Ψ:    {rmse_total:.4e}")
    print(f"  RMSE per-Ψ-comp:")
    for nm, r in zip(psi_names, rmse_per_psi):
        print(f"     {nm:>3s}: {r:.4e}")
    # Original-state RMSE (first 4 components of Ψ).
    rmse_native = np.sqrt(np.mean(resid[:, :4] ** 2, axis=0))
    rmse_native_total = float(np.sqrt(np.mean(np.sum(resid[:, :4] ** 2, axis=1))))
    print(f"  RMSE on native state z:")
    for nm, r in zip(["θ1","θ2","ω1","ω2"], rmse_native):
        print(f"     {nm:>3s}: {r:.4e}")
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
    out_path = os.path.join(results_dir, "walkthrough_pendulum_path_i.json")

    print("=" * 72)
    print(" Walkthrough smoke test — Pendulum path (i): Koopman augmentation")
    print("=" * 72)

    z0 = np.array([np.pi - 0.1, np.pi - 0.1, 0.0, 0.0])

    configs = [
        ("dt=1e-2, T=5000",  1e-2, 5000),
        ("dt=1e-3, T=5000",  1e-3, 5000),
        ("dt=1e-4, T=5000",  1e-4, 5000),
        ("dt=1e-3, T=20000", 1e-3, 20000),
    ]

    all_results = {"runs": []}
    for label, dt, T in configs:
        t0 = time.time()
        Z = euler_integrate(z0, dt, T)
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
