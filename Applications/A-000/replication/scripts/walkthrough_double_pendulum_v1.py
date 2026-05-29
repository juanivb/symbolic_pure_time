"""
Walkthrough — double pendulum smoke test (path ii: enriched basis).

Equal-mass equal-length compound double pendulum (m1=m2=L1=L2=1, g=9.81).
State: z = (θ1, θ2, ω1, ω2).

EOM (Lagrangian formulation, simplified for unit masses & lengths):

    ω̇1 =  [-3 g sin θ1
            -   g sin(θ1 - 2 θ2)
            - 2 sin(θ1 - θ2) ( ω2^2 + ω1^2 cos(θ1 - θ2) ) ] / D

    ω̇2 = [ 2 sin(θ1 - θ2) ( 2 ω1^2 + 2 g cos θ1 + ω2^2 cos(θ1 - θ2) ) ] / D

    D  = 3 - cos(2(θ1 - θ2))

Atomic terms in the numerators (with φ = θ1 - θ2):
    sin θ1
    sin(θ1 - 2 θ2)
    sin φ · ω2^2
    sin φ · cos φ · ω1^2
    sin φ · ω1^2
    sin φ · cos θ1
    sin φ · cos φ · ω2^2

Path (ii) feature library  =  { pure-time linear } ∪ { atoms } ∪ { atoms / D }.

Forward Euler discretisation matches the Lorenz example so the OLS target
is z_{t+1} = z_t + dt · f(z_t) and the recovery is purely a function of
basis adequacy.
"""
from __future__ import annotations

import json
import os
import time

import numpy as np


G = 9.81


# ---------------------------------------------------------------------
# Dynamics
# ---------------------------------------------------------------------

def pendulum_rhs(z, g=G):
    th1, th2, w1, w2 = z
    phi = th1 - th2
    s_phi = np.sin(phi); c_phi = np.cos(phi)
    s2phi_factor = np.cos(2 * phi)
    D = 3.0 - s2phi_factor
    num1 = (-3 * g * np.sin(th1)
            - g * np.sin(th1 - 2 * th2)
            - 2 * s_phi * (w2 * w2 + w1 * w1 * c_phi))
    num2 = (2 * s_phi * (2 * w1 * w1
                          + 2 * g * np.cos(th1)
                          + w2 * w2 * c_phi))
    return np.array([w1, w2, num1 / D, num2 / D])


def euler_integrate(z0, dt, T):
    Z = np.empty((T, 4))
    Z[0] = z0
    for t in range(1, T):
        Z[t] = Z[t-1] + dt * pendulum_rhs(Z[t-1])
    return Z


def rk4_integrate(z0, dt, T):
    Z = np.empty((T, 4)); Z[0] = z0
    for t in range(1, T):
        z = Z[t-1]
        k1 = pendulum_rhs(z)
        k2 = pendulum_rhs(z + 0.5 * dt * k1)
        k3 = pendulum_rhs(z + 0.5 * dt * k2)
        k4 = pendulum_rhs(z + dt * k3)
        Z[t] = z + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)
    return Z


def hamiltonian(z, g=G):
    """Total energy. m=L=1 for both segments.

    H = T + V with
        T = (1/2)(2 ω1² + ω2² + 2 ω1 ω2 cos φ)
        V = -g (2 cos θ1 + cos θ2)
    """
    th1, th2, w1, w2 = z
    phi = th1 - th2
    T = 0.5 * (2 * w1 * w1 + w2 * w2 + 2 * w1 * w2 * np.cos(phi))
    V = -g * (2 * np.cos(th1) + np.cos(th2))
    return float(T + V)


# ---------------------------------------------------------------------
# Feature library — path (ii)
# ---------------------------------------------------------------------

def enriched_features(Z):
    """Build features from rows z = (θ1, θ2, ω1, ω2)."""
    th1 = Z[:, 0]; th2 = Z[:, 1]; w1 = Z[:, 2]; w2 = Z[:, 3]
    phi = th1 - th2
    s_phi = np.sin(phi); c_phi = np.cos(phi)
    s_2phi = np.sin(2 * phi); c_2phi = np.cos(2 * phi)
    D = 3.0 - c_2phi
    invD = 1.0 / D

    # Atoms appearing in the EOM numerators (and a few extras for safety).
    atoms = {
        "sin_th1":           np.sin(th1),
        "cos_th1":           np.cos(th1),
        "sin_th2":           np.sin(th2),
        "cos_th2":           np.cos(th2),
        "sin_phi":           s_phi,
        "cos_phi":           c_phi,
        "sin_phi_cos_phi":   s_phi * c_phi,
        "sin_th1m2th2":      np.sin(th1 - 2 * th2),
        "sin_phi_w1sq":      s_phi * w1 * w1,
        "sin_phi_w2sq":      s_phi * w2 * w2,
        "sin_phi_cos_phi_w1sq":  s_phi * c_phi * w1 * w1,
        "sin_phi_cos_phi_w2sq":  s_phi * c_phi * w2 * w2,
        "sin_phi_cos_th1":   s_phi * np.cos(th1),
        "sin_phi_cos_th2":   s_phi * np.cos(th2),
        "sin_phi_w1w2":      s_phi * w1 * w2,
        "cos_phi_w1sq":      c_phi * w1 * w1,
        "cos_phi_w2sq":      c_phi * w2 * w2,
    }

    cols = []; names = []
    # Group A: pure-time linear (z_t and Δz_t added later in build_design)
    # Group B: raw atoms (numerators are polynomials in atoms)
    for nm, col in atoms.items():
        cols.append(col); names.append(nm)
    # Group C: atoms / D — these are the rational terms that span N/D directly
    for nm, col in atoms.items():
        cols.append(col * invD); names.append(nm + "/D")
    # Group D: 1/D itself (in case there is a constant additive term / D)
    cols.append(invD); names.append("1/D")
    return np.column_stack(cols), names


def build_design(Z):
    z_t = Z[1:-1]
    dz_t = Z[1:-1] - Z[:-2]
    z_next = Z[2:]
    enr, enr_names = enriched_features(z_t)
    X = np.hstack([z_t, dz_t, enr])
    names = (["z_th1", "z_th2", "z_w1", "z_w2",
              "dz_th1", "dz_th2", "dz_w1", "dz_w2"]
             + enr_names)
    return X, z_next, names


# ---------------------------------------------------------------------
# Fit + report
# ---------------------------------------------------------------------

def fit_evaluate(Z, label):
    X, Y, names = build_design(Z)
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
    comp_names = ["th1", "th2", "w1", "w2"]
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


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    results_dir = os.path.normpath(os.path.join(here, "..", "results"))
    os.makedirs(results_dir, exist_ok=True)
    out_path = os.path.join(results_dir, "walkthrough_double_pendulum_v1.json")

    print("=" * 72)
    print(" Walkthrough smoke test — Double pendulum, path (ii) enriched basis")
    print("=" * 72)

    # Initial condition: well-known chaotic regime.
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
        H0 = hamiltonian(Z[0]); HT = hamiltonian(Z[-1])
        rel_drift = (HT - H0) / abs(H0)
        print(f"\n[Euler  {label}] integrated in {time.time()-t0:.2f}s; "
              f"H drift = {rel_drift:+.3e}")
        res = fit_evaluate(Z, f"Euler {label}")
        res["dt"] = dt; res["T"] = T
        res["energy_drift_relative"] = float(rel_drift)
        res["integrator"] = "euler"
        all_results["runs"].append(res)

    # RK4 sanity check.
    t0 = time.time()
    Z_rk4 = rk4_integrate(z0, dt=1e-3, T=5000)
    H0 = hamiltonian(Z_rk4[0]); HT = hamiltonian(Z_rk4[-1])
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
