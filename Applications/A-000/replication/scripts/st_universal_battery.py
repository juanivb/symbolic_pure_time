"""
ST-Universal — Battery for the universal univariate estimator
across regimes (stationary / non-stationary / explosive).

Demonstrates points (i) and (iii) of Juani's prompt:
- (i) Comparison vs state-of-the-art baselines (OLS-AR(2), Yule-Walker)
- (iii) Transparency / generality: same estimator across regimes.

DGPs tested:
  R1: AR(1) stable      φ = 0.5
  R2: AR(1) near-unit   φ = 0.95
  R3: Random walk       φ = 1.0   (non-stationary I(1))
  R4: AR(1) explosive   φ = 1.05  (explosive)
  R5: AR(2) stable      (φ_1, φ_2) = (1.4, -0.6)
  R6: AR(2) near-unit   (φ_1, φ_2) = (0.9, 0.05) — close to unit root
  R7: Sum of 2 sines + small noise  (rank-3 quasi-periodic)
  R8: I(2) (double random walk)

Each cell: T = 1000, N = 100 seeds. Report bias, std, MSE of
parameter recovery in (φ_1, φ_2) space.
"""
from __future__ import annotations

import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from action_graded_universal import (
    ptls_universal_level1, ar2_ols, yule_walker_ar2,
    ab_to_phi, phi_to_ab,
)

T = 1000
N_REPS = 100
SEED = 42

PROJECT_ROOT = os.path.abspath(os.path.join(HERE, ".."))
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")
os.makedirs(RESULTS_DIR, exist_ok=True)


# ---------------------------------------------------------------------
# DGPs
# ---------------------------------------------------------------------

def gen_ar2(phi1, phi2, T, sigma, rng):
    eps = sigma * rng.standard_normal(T)
    y = np.empty(T)
    y[0] = eps[0]
    if T > 1:
        y[1] = phi1 * y[0] + eps[1]
    for t in range(2, T):
        y[t] = phi1 * y[t - 1] + phi2 * y[t - 2] + eps[t]
    return y


def gen_ar1(phi, T, sigma, rng):
    return gen_ar2(phi, 0.0, T, sigma, rng)


def gen_rw(T, sigma, rng):
    return np.cumsum(sigma * rng.standard_normal(T))


def gen_i2(T, sigma, rng):
    return np.cumsum(np.cumsum(sigma * rng.standard_normal(T)))


def gen_sum2sines(T, sigma, rng):
    t = np.arange(T)
    return (np.sin(0.3 * t + 0.5) + 0.7 * np.sin(0.7 * t)
            + sigma * rng.standard_normal(T))


# ---------------------------------------------------------------------
# Cells
# ---------------------------------------------------------------------

CELLS = [
    # (label, generator_fn, true_(phi1, phi2), σ_default)
    ("R1: AR(1) φ=0.5",       lambda T, sg, r: gen_ar1(0.5, T, sg, r),  (0.5, 0.0),    1.0),
    ("R2: AR(1) φ=0.95",      lambda T, sg, r: gen_ar1(0.95, T, sg, r), (0.95, 0.0),   1.0),
    ("R3: Random walk",       lambda T, sg, r: gen_rw(T, sg, r),        (1.0, 0.0),    1.0),
    ("R4: AR(1) φ=1.05",      lambda T, sg, r: gen_ar1(1.05, T, sg, r), (1.05, 0.0),   1.0),
    ("R5: AR(2) (1.4,-0.6)",  lambda T, sg, r: gen_ar2(1.4, -0.6, T, sg, r), (1.4, -0.6), 1.0),
    ("R6: AR(2) (0.9,0.05)",  lambda T, sg, r: gen_ar2(0.9, 0.05, T, sg, r), (0.9, 0.05), 1.0),
    ("R7: 2-sines + noise",   lambda T, sg, r: gen_sum2sines(T, sg, r), (None, None),  0.05),
    ("R8: I(2)",              lambda T, sg, r: gen_i2(T, sg, r),         (None, None), 1.0),
]


def run_cell(label, gen_fn, phi_true, sigma):
    rng_master = np.random.default_rng(SEED)
    seeds = rng_master.integers(0, 2**31, size=N_REPS)
    rows = []
    for s in seeds:
        rng = np.random.default_rng(int(s))
        y = gen_fn(T, sigma, rng)

        # PTLS-Universal (level-1, closed form)
        a, b = ptls_universal_level1(y)
        phi1_pt, phi2_pt = ab_to_phi(a, b)

        # OLS-AR(2)
        phi1_ols, phi2_ols = ar2_ols(y)

        # Yule-Walker AR(2)
        phi1_yw, phi2_yw = yule_walker_ar2(y)

        rows.append((phi1_pt, phi2_pt, phi1_ols, phi2_ols, phi1_yw, phi2_yw, a, b))
    return np.array(rows)


def main():
    print("=" * 100)
    print(f"ST-Universal — universal estimator vs OLS-AR(2) vs Yule-Walker | "
          f"T={T}, N={N_REPS}")
    print("=" * 100)

    print(f"\n{'cell':>26s} | {'φ_1 true':>10s} | "
          f"{'PT-univ φ̂_1':>14s} {'OLS-AR φ̂_1':>14s} {'YW φ̂_1':>14s} | "
          f"{'time':>6s}")

    summary = []
    for label, gen_fn, (p1, p2), sigma in CELLS:
        t0 = time.time()
        arr = run_cell(label, gen_fn, (p1, p2), sigma)
        elapsed = time.time() - t0

        means = arr.mean(axis=0)
        stds = arr.std(axis=0, ddof=1)

        if p1 is not None:
            mse_pt = float(((arr[:, 0] - p1) ** 2 + (arr[:, 1] - p2) ** 2).mean())
            mse_ols = float(((arr[:, 2] - p1) ** 2 + (arr[:, 3] - p2) ** 2).mean())
            mse_yw = float(((arr[:, 4] - p1) ** 2 + (arr[:, 5] - p2) ** 2).mean())
        else:
            mse_pt = mse_ols = mse_yw = float("nan")

        p1_str = f"{p1:.3f}" if p1 is not None else "—"
        print(f"{label:>26s} | {p1_str:>10s} | "
              f"{means[0]:>+9.4f}({stds[0]:.2f}) {means[2]:>+9.4f}({stds[2]:.2f}) "
              f"{means[4]:>+9.4f}({stds[4]:.2f}) | {elapsed:>5.2f}s")

        summary.append({
            "cell": label, "phi1_true": p1, "phi2_true": p2,
            "PT_phi1_mean": means[0], "PT_phi2_mean": means[1],
            "PT_phi1_std": stds[0], "PT_phi2_std": stds[1],
            "PT_alpha_mean": means[6], "PT_beta_mean": means[7],
            "OLS_phi1_mean": means[2], "OLS_phi2_mean": means[3],
            "YW_phi1_mean": means[4], "YW_phi2_mean": means[5],
            "MSE_PT": mse_pt, "MSE_OLS": mse_ols, "MSE_YW": mse_yw,
        })

    print()
    print(f"{'cell':>26s} | {'MSE PT-univ':>14s} {'MSE OLS-AR':>14s} {'MSE YW':>14s}")
    for s in summary:
        if not np.isnan(s["MSE_PT"]):
            print(f"{s['cell']:>26s} | {s['MSE_PT']:>14.6e} "
                  f"{s['MSE_OLS']:>14.6e} {s['MSE_YW']:>14.6e}")
        else:
            print(f"{s['cell']:>26s} | (no φ ground truth — universal applies)")
            # Print the (α, β) estimates as the universal parametrisation.
            print(f"                              (α̂, β̂) = "
                  f"({s['PT_alpha_mean']:.4f}, {s['PT_beta_mean']:.4f})")

    out_csv = os.path.join(RESULTS_DIR, "st_universal_battery.csv")
    import csv as csv_mod
    with open(out_csv, "w", newline="") as f:
        keys = list(summary[0].keys())
        w = csv_mod.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for row in summary:
            w.writerow(row)
    print(f"\nWrote {out_csv}")


if __name__ == "__main__":
    main()
