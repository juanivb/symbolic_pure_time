"""
Walkthrough — three-body smoke test (path ii: enriched basis).

Tests whether OLS in the pure-time basis enriched with explicit Newtonian
gravitational features recovers the Euler-discretised three-body dynamics
to machine precision, in analogy to Lorenz '63 with the quadratic basis.

System: planar three-body, equal masses, G = 1.
Configuration: figure-eight (Chenciner-Montgomery 2000) — stable periodic
orbit, well-conditioned for the smoke test.

State: 12-d, ordered as
    z = (r1x, r1y, r2x, r2y, r3x, r3y, v1x, v1y, v2x, v2y, v3x, v3y).

Discretisation: forward Euler with dt small enough that the orbit does
not drift catastrophically over the simulation horizon. Forward Euler
matches the Lorenz example exactly so that recovery is purely a function
of basis adequacy.

Feature library:
  * pure-time linear:  z_t (12)  and  Δz_t = z_t - z_{t-1} (12)
  * gravitational:     for each ordered pair (i, j) with i ≠ j,
                       g_{ij}(z_t) = (r_j - r_i) / |r_j - r_i|^3   (2 comps)
                       — 6 pairs × 2 = 12 features

Target: z_{t+1} (12 components, fit per-component by OLS).

Reports per-component RMSE in-sample. Comparison reference: Lorenz with
quadratic basis hits ~1e-13.
"""
from __future__ import annotations

import json
import os
import sys
import time

import numpy as np


# ---------------------------------------------------------------------
# Three-body planar dynamics
# ---------------------------------------------------------------------

def three_body_rhs(z, masses, G=1.0):
    """RHS of the 12-d state ODE.

    Parameters
    ----------
    z : array, shape (12,)
        State vector ordered (r1, r2, r3, v1, v2, v3) with each
        position/velocity 2-d.
    masses : array, shape (3,)
    G : float

    Returns
    -------
    dz : array, shape (12,)
    """
    r = z[:6].reshape(3, 2)
    v = z[6:].reshape(3, 2)
    a = np.zeros_like(r)
    for i in range(3):
        for j in range(3):
            if i == j:
                continue
            d = r[j] - r[i]
            r3 = (d @ d) ** 1.5
            a[i] += G * masses[j] * d / r3
    return np.concatenate([v.ravel(), a.ravel()])


def euler_integrate(z0, masses, G, dt, T):
    """Forward Euler (matches the Lorenz example's discretisation)."""
    Z = np.empty((T, len(z0)))
    Z[0] = z0
    for t in range(1, T):
        Z[t] = Z[t-1] + dt * three_body_rhs(Z[t-1], masses, G)
    return Z


def rk4_integrate(z0, masses, G, dt, T):
    """RK4 integrator — for sanity check on energy drift."""
    Z = np.empty((T, len(z0)))
    Z[0] = z0
    f = lambda zz: three_body_rhs(zz, masses, G)
    for t in range(1, T):
        z = Z[t-1]
        k1 = f(z)
        k2 = f(z + 0.5 * dt * k1)
        k3 = f(z + 0.5 * dt * k2)
        k4 = f(z + dt * k3)
        Z[t] = z + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)
    return Z


def hamiltonian(z, masses, G):
    """Total energy H = T + V (for diagnostics)."""
    r = z[:6].reshape(3, 2)
    v = z[6:].reshape(3, 2)
    T = 0.5 * np.sum(masses[:, None] * v * v)
    V = 0.0
    for i in range(3):
        for j in range(i + 1, 3):
            d = r[j] - r[i]
            V -= G * masses[i] * masses[j] / np.sqrt(d @ d)
    return float(T + V)


# ---------------------------------------------------------------------
# Initial condition: Chenciner-Montgomery figure-eight
# ---------------------------------------------------------------------

def figure_eight_ic():
    # Standard rationalised values from Chenciner-Montgomery (2000).
    # Three equal masses (m=1), G=1, period ~ 6.32591398.
    p1 = np.array([0.97000436, -0.24308753])
    p3 = -p1                          # body 3 mirrors body 1
    p2 = np.zeros(2)                  # body 2 at origin
    v3 = np.array([-0.93240737, -0.86473146])
    v1 = -v3 / 2.0
    v2 = -v3 / 2.0
    z0 = np.concatenate([p1, p2, p3, v1, v2, v3])
    return z0


# ---------------------------------------------------------------------
# Feature library — path (ii): pure-time + gravitational
# ---------------------------------------------------------------------

def gravitational_features(Z):
    """For each row z of Z, compute 12 gravitational unit-force features.

    Pair (i, j), j ≠ i, contributes 2 features: (r_j - r_i) / |r_j - r_i|^3.
    Order: (0,1), (0,2), (1,0), (1,2), (2,0), (2,1).
    Returns array shape (T, 12).
    """
    T = Z.shape[0]
    R = Z[:, :6].reshape(T, 3, 2)
    feats = []
    for i in range(3):
        for j in range(3):
            if i == j:
                continue
            d = R[:, j] - R[:, i]                   # (T, 2)
            r3 = (np.sum(d * d, axis=1) ** 1.5)[:, None]
            feats.append(d / r3)
    return np.hstack(feats)                          # (T, 12)


def build_design(Z):
    """Pure-time + gravitational features.

    Returns (X, Y_next) where X has rows (z_t, Δz_t, gravitational(z_t))
    and Y_next has rows z_{t+1}. Indexing aligned with the quad_design_mv
    convention of the published examples.
    """
    T = Z.shape[0]
    z_t = Z[1:-1]                                   # (T-2, 12)
    dz_t = Z[1:-1] - Z[:-2]                         # (T-2, 12)
    z_next = Z[2:]                                  # (T-2, 12)
    g_t = gravitational_features(z_t)               # (T-2, 12)
    X = np.hstack([z_t, dz_t, g_t])                 # (T-2, 36)
    return X, z_next


# ---------------------------------------------------------------------
# Fit + evaluate
# ---------------------------------------------------------------------

def fit_evaluate(Z, label):
    X, Y = build_design(Z)
    coef, *_ = np.linalg.lstsq(X, Y, rcond=None)
    pred = X @ coef
    resid = pred - Y
    rmse_per = np.sqrt(np.mean(resid * resid, axis=0))
    rmse_total = float(np.sqrt(np.mean(np.sum(resid * resid, axis=1))))
    cond_X = float(np.linalg.cond(X))
    print(f"\n=== {label} ===")
    print(f"  T_samples used:  {X.shape[0]}")
    print(f"  features:        {X.shape[1]}")
    print(f"  cond(X):         {cond_X:.3e}")
    print(f"  RMSE total:      {rmse_total:.4e}")
    print(f"  RMSE per-comp:")
    comp_names = ["r1x","r1y","r2x","r2y","r3x","r3y",
                  "v1x","v1y","v2x","v2y","v3x","v3y"]
    for nm, r in zip(comp_names, rmse_per):
        print(f"     {nm:>4s}: {r:.4e}")
    return {
        "label": label,
        "T_samples": int(X.shape[0]),
        "n_features": int(X.shape[1]),
        "cond_X": cond_X,
        "rmse_total": rmse_total,
        "rmse_per_comp": {nm: float(r) for nm, r in zip(comp_names, rmse_per)},
    }


def compare_lorenz_baseline():
    """Recompute the Lorenz baseline RMSE for reference, using the same
    code path (Euler integration, OLS on a polynomial basis that contains
    the truth)."""
    sigma_l, rho_l, beta_l = 10.0, 28.0, 8.0/3.0
    dt, T = 0.01, 5000
    Y = np.empty((T, 3)); Y[0] = (1, 1, 1)
    for t in range(1, T):
        x, y, z = Y[t-1]
        Y[t, 0] = x + dt * sigma_l * (y - x)
        Y[t, 1] = y + dt * (x * (rho_l - z) - y)
        Y[t, 2] = z + dt * (x * y - beta_l * z)
    # Quadratic basis (matches the published example's quad_design_mv).
    Y_t = Y[1:-1]; dY_t = Y[1:-1] - Y[:-2]; Y_next = Y[2:]
    feats = [np.ones((Y_t.shape[0], 1)), Y_t, dY_t,
             Y_t * Y_t, dY_t * dY_t]
    for j in range(3):
        for k in range(j+1, 3):
            feats.append((Y_t[:, j] * Y_t[:, k]).reshape(-1, 1))
    for j in range(3):
        for k in range(j+1, 3):
            feats.append((dY_t[:, j] * dY_t[:, k]).reshape(-1, 1))
    for j in range(3):
        for k in range(3):
            feats.append((Y_t[:, j] * dY_t[:, k]).reshape(-1, 1))
    X = np.hstack(feats)
    coef, *_ = np.linalg.lstsq(X, Y_next, rcond=None)
    pred = X @ coef
    rmse = float(np.sqrt(np.mean(np.sum((pred - Y_next) ** 2, axis=1))))
    return rmse


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    results_dir = os.path.normpath(os.path.join(here, "..", "results"))
    os.makedirs(results_dir, exist_ok=True)
    out_path = os.path.join(results_dir, "walkthrough_three_body_v1.json")

    print("=" * 72)
    print(" Walkthrough smoke test — Three-body, path (ii) enriched basis")
    print("=" * 72)

    # Lorenz baseline (sanity check that we reproduce ~1e-13).
    t0 = time.time()
    lorenz_rmse = compare_lorenz_baseline()
    print(f"\nLorenz baseline (Euler, quadratic basis):  RMSE = {lorenz_rmse:.4e}"
          f"   ({time.time()-t0:.2f}s)")

    masses = np.array([1.0, 1.0, 1.0])
    G = 1.0
    z0 = figure_eight_ic()

    # --- Configuration sweep on dt and T ---------------------------------
    configs = [
        ("dt=1e-2, T=5000",  1e-2, 5000),
        ("dt=1e-3, T=5000",  1e-3, 5000),
        ("dt=1e-4, T=5000",  1e-4, 5000),
        ("dt=1e-3, T=20000", 1e-3, 20000),
    ]

    all_results = {"lorenz_baseline_rmse": lorenz_rmse, "runs": []}
    for label, dt, T in configs:
        t0 = time.time()
        Z = euler_integrate(z0, masses, G, dt, T)
        # Energy drift diagnostic
        H0 = hamiltonian(Z[0], masses, G)
        HT = hamiltonian(Z[-1], masses, G)
        rel_drift = (HT - H0) / abs(H0)
        print(f"\n[Euler  {label}] integrated in {time.time()-t0:.2f}s; "
              f"H drift = {rel_drift:+.3e}")
        res = fit_evaluate(Z, f"Euler {label}")
        res["dt"] = dt; res["T"] = T
        res["energy_drift_relative"] = float(rel_drift)
        res["integrator"] = "euler"
        all_results["runs"].append(res)

    # --- Same with RK4 for one config, to see if higher-order discretisation
    # breaks the recovery (it should, partially) ----------------------------
    t0 = time.time()
    Z_rk4 = rk4_integrate(z0, masses, G, dt=1e-3, T=5000)
    H0 = hamiltonian(Z_rk4[0], masses, G); HT = hamiltonian(Z_rk4[-1], masses, G)
    print(f"\n[RK4   dt=1e-3, T=5000] integrated in {time.time()-t0:.2f}s; "
          f"H drift = {(HT-H0)/abs(H0):+.3e}")
    res = fit_evaluate(Z_rk4, "RK4 dt=1e-3, T=5000")
    res["dt"] = 1e-3; res["T"] = 5000
    res["energy_drift_relative"] = float((HT-H0)/abs(H0))
    res["integrator"] = "rk4"
    all_results["runs"].append(res)

    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to: {out_path}")


if __name__ == "__main__":
    main()
