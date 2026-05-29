"""
ST-Lorenz — Multivariate quadratic estimator on Lorenz '63 chaos.

Demonstrates that the multivariate pure-time quadratic estimator
recovers the 3D Lorenz dynamics from data, including the bilinear
cross-terms x·z and x·y that classical VAR methods cannot capture.

Lorenz Euler-discretised:
   x_{t+1} = (1 - dt σ) x_t + dt σ y_t
   y_{t+1} = dt ρ x_t + (1 - dt) y_t - dt x_t z_t
   z_{t+1} = (1 - dt β) z_t + dt x_t y_t

So the expected coefficients per equation are:
   eq x:  intercept 0, linear x = 1 - dt σ = 0.9, linear y = dt σ = 0.1,
          all other = 0.
   eq y:  intercept 0, linear x = dt ρ = 0.28, linear y = 1 - dt = 0.99,
          cross x z = -dt = -0.01, all other = 0.
   eq z:  intercept 0, linear z = 1 - dt β = 0.97333, cross x y = dt = 0.01,
          all other = 0.

(Standard params: σ=10, ρ=28, β=8/3, dt=0.01.)
"""
from __future__ import annotations

import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from action_graded_mv_quadratic import (
    ptmv_quadratic, feature_names, lorenz_euler, coupled_logistic, rossler_euler,
)
from action_graded_multivariate import ptmv_level1 as ptmv, var2_ols

PROJECT_ROOT = os.path.abspath(os.path.join(HERE, ".."))
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")
os.makedirs(RESULTS_DIR, exist_ok=True)


def report_recovery(label, Y, expected_dict=None):
    """Run multivariate quadratic OLS and report top coefficients."""
    coef = ptmv_quadratic(Y)
    names = feature_names(Y.shape[1])
    d = Y.shape[1]
    print(f"\n{'='*70}\n  {label}  |  shape = {Y.shape}\n{'='*70}")

    # Pretty-print coefficients above threshold per equation
    threshold = 1e-3
    for i in range(d):
        print(f"\n  Equation y_{i+1}_{{t+1}} (coeffs |c|>{threshold:.0e}):")
        for f, c_val in zip(names, coef[:, i]):
            if abs(c_val) > threshold:
                marker = ""
                if expected_dict is not None and i in expected_dict:
                    truth = expected_dict[i].get(f, 0.0)
                    err = abs(c_val - truth)
                    marker = f"  [truth {truth:+.5f}, |Δ| = {err:.2e}]"
                print(f"    {f:>14s}  =  {c_val:>+12.6f}{marker}")

    # Compare with linear PT-MV (baseline)
    A, B = ptmv(Y)
    print(f"\n  Linear PT-MV baseline (A, B) shapes: {A.shape}, {B.shape}")
    # Also linear forecast residual
    Y_t = Y[1:-1]; dY_t = Y[1:-1] - Y[:-2]; Y_next = Y[2:]
    pred_lin = Y_t @ A.T + dY_t @ B.T
    err_lin = float(np.sqrt(np.mean(np.sum((pred_lin - Y_next) ** 2, axis=1))))

    # And quadratic forecast residual
    from action_graded_mv_quadratic import _quad_design_mv
    X, Y_next = _quad_design_mv(Y)
    pred_q = X @ coef
    err_q = float(np.sqrt(np.mean(np.sum((pred_q - Y_next) ** 2, axis=1))))

    print(f"\n  In-sample RMSE per step:")
    print(f"     Linear PT-MV:     {err_lin:.6e}")
    print(f"     Quadratic PT-MV:  {err_q:.6e}")
    print(f"     Ratio quad/lin:   {err_q / max(err_lin, 1e-300):.4e}")
    return coef, err_lin, err_q


def main():
    print("=" * 70)
    print(" ST-Lorenz — Multivariate quadratic estimator on chaos")
    print("=" * 70)

    # 1) Lorenz, noise-free
    sigma, rho, beta, dt, T = 10.0, 28.0, 8.0/3.0, 0.01, 5000
    Y = lorenz_euler(T, dt=dt, sigma=sigma, rho=rho, beta=beta)
    expected = {
        0: {"y_1": 1 - dt*sigma, "y_2": dt*sigma},
        1: {"y_1": dt*rho, "y_2": 1 - dt, "y_1*y_3": -dt},
        2: {"y_3": 1 - dt*beta, "y_1*y_2": dt},
    }
    report_recovery(f"LORENZ '63 (σ={sigma}, ρ={rho}, β={beta:.3f}, dt={dt})",
                       Y, expected)

    # 2) Lorenz with noise
    rng = np.random.default_rng(42)
    Y_noisy = Y + 0.05 * rng.standard_normal(Y.shape)
    print("\n--- Lorenz + noise σ=0.05 ---")
    report_recovery("LORENZ noisy (σ_noise=0.05)", Y_noisy, expected)

    # 3) Coupled logistic maps
    Yc = coupled_logistic(T, r=3.95, eps_couple=0.05)
    print("\n--- Coupled logistic (r=3.95, ε=0.05) ---")
    report_recovery("COUPLED LOGISTIC", Yc, expected_dict=None)

    # 4) Rössler system
    Yr = rossler_euler(T, dt=0.05, a=0.2, b=0.2, c=5.7)
    print("\n--- Rössler (a=0.2, b=0.2, c=5.7) ---")
    report_recovery("RÖSSLER", Yr, expected_dict=None)


if __name__ == "__main__":
    main()
