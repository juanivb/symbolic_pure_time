"""
Walkthrough — SINDy in OLS baseline.

Tests how well a vanilla SINDy-style regression — polynomial dictionary
of degree d in pure z_t (no pure-time Δ, no Koopman augmentation, no
system-aware enrichment) — recovers the dynamics of pendulum and
three-body. Provides the head-to-head comparison against PTLS path (i)
and path (ii) for the divulgation chapter.

For each system and each dictionary degree d, we fit:
    ż  ≈  poly_d(z)
where poly_d(z) is the full multivariate polynomial up to degree d on
the original state z (no augmented observables), and ż is approximated
by  (z_{t+1} - z_t) / dt  from an Euler-discretised trajectory.
"""
from __future__ import annotations

import json
import os
from itertools import combinations_with_replacement

import numpy as np


# ---------------------------------------------------------------------
# Common dictionaries
# ---------------------------------------------------------------------

def poly_features(Z, degree):
    """Multivariate polynomial features up to total degree ``degree``.

    Z : (T, d) state.
    Returns (T, K) feature matrix and the list of feature names.
    """
    T, d = Z.shape
    feats = [np.ones(T)]
    names = ["1"]
    for k in range(1, degree + 1):
        for combo in combinations_with_replacement(range(d), k):
            col = np.ones(T)
            for idx in combo:
                col = col * Z[:, idx]
            feats.append(col)
            names.append("·".join(f"z{i}" for i in combo))
    return np.column_stack(feats), names


def fit_sindy(Z, dt, degree):
    """Fit SINDy polynomial dictionary of given degree on Euler-derivative
    target. Returns (RMSE_per_component, n_features)."""
    Z_t = Z[:-1]
    dz = (Z[1:] - Z[:-1]) / dt        # Euler approximation of ż
    X, names = poly_features(Z_t, degree)
    coef, *_ = np.linalg.lstsq(X, dz, rcond=None)
    pred = X @ coef
    resid = pred - dz
    rmse_per = np.sqrt(np.mean(resid * resid, axis=0))
    rmse_total = float(np.sqrt(np.mean(np.sum(resid * resid, axis=1))))
    return rmse_per, rmse_total, X.shape[1]


# ---------------------------------------------------------------------
# Pendulum
# ---------------------------------------------------------------------

G = 9.81


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


def integrate_pendulum(z0, dt, T):
    Z = np.empty((T, 4)); Z[0] = z0
    for t in range(1, T):
        Z[t] = Z[t-1] + dt * pendulum_rhs(Z[t-1])
    return Z


# ---------------------------------------------------------------------
# Three-body
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


def integrate_threebody(z0, masses, G, dt, T):
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
# Driver
# ---------------------------------------------------------------------

def main():
    here = os.path.dirname(os.path.abspath(__file__))
    results_dir = os.path.normpath(os.path.join(here, "..", "results"))
    os.makedirs(results_dir, exist_ok=True)
    out_path = os.path.join(results_dir, "walkthrough_sindy_baseline.json")

    print("=" * 72)
    print(" Walkthrough — SINDy in OLS baseline (no pure-time, no Koopman)")
    print("=" * 72)

    all_results = {}

    # --- Pendulum ---
    print("\n--- Pendulum (4-d state) ---")
    z0 = np.array([np.pi - 0.1, np.pi - 0.1, 0.0, 0.0])
    Z = integrate_pendulum(z0, 1e-3, 5000)
    pendulum_runs = []
    for d in [1, 2, 3, 4, 5]:
        rmse_per, rmse_total, nfeat = fit_sindy(Z, dt=1e-3, degree=d)
        print(f"  degree={d}, n_features={nfeat:4d}, RMSE_total = {rmse_total:.4e}")
        pendulum_runs.append({
            "degree": d, "n_features": nfeat,
            "rmse_total": rmse_total,
            "rmse_per": [float(x) for x in rmse_per],
        })
    all_results["pendulum"] = {
        "dt": 1e-3, "T": 5000,
        "ic": z0.tolist(),
        "runs": pendulum_runs,
    }

    # --- Three-body ---
    print("\n--- Three-body (12-d state, figure-eight IC) ---")
    masses = np.array([1.0, 1.0, 1.0]); G = 1.0
    z0 = figure_eight_ic()
    Z = integrate_threebody(z0, masses, G, 1e-4, 5000)
    threebody_runs = []
    for d in [1, 2, 3, 4]:
        rmse_per, rmse_total, nfeat = fit_sindy(Z, dt=1e-4, degree=d)
        print(f"  degree={d}, n_features={nfeat:5d}, RMSE_total = {rmse_total:.4e}")
        threebody_runs.append({
            "degree": d, "n_features": nfeat,
            "rmse_total": rmse_total,
            "rmse_per": [float(x) for x in rmse_per],
        })
    all_results["three_body"] = {
        "dt": 1e-4, "T": 5000,
        "ic": z0.tolist(),
        "runs": threebody_runs,
    }

    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to: {out_path}")


if __name__ == "__main__":
    main()
