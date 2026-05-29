"""
ST-Multivariate — Battery for the multivariate pure-time estimator
across multivariate regimes.

DGPs:
  M1: Two independent AR(1)s          (rank(M)=2 → no cointegration)
  M2: Cointegrated I(1) pair          (rank(M)=1 → 1 cointegrating vector)
  M3: Triangular VAR (one-way causal) (stationary)
  M4: Mixed I(1) + AR(1) stationary
  M5: Bivariate explosive/stationary  (one explosive, one AR(1))
  M6: Two correlated random walks (no cointegration; rank(M)=0)
  M7: VAR(1) full coupling stationary

For each:
  - Recover (A, B) → derived (Φ_1, Φ_2) via PT-MV
  - Recover (Φ_1, Φ_2) via classical VAR(2)
  - For bivariate I(1) case (M2), also try VECM (Engle-Granger 2-step)
  - Report parameter MSE plus 1-step forecast RMSE
  - Report estimated rank of error-correction matrix M = A + B - I
"""
from __future__ import annotations

import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from action_graded_multivariate import (
    ptmv_level1, AB_to_Phi, ec_matrix, emergent_rank,
    var2_ols, vecm_engle_granger_2step,
)

T = 800
N_REPS = 30
SEED = 42

PROJECT_ROOT = os.path.abspath(os.path.join(HERE, ".."))
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")
os.makedirs(RESULTS_DIR, exist_ok=True)


# ---------------------------------------------------------------------
# DGPs (each returns (Y, true_(Phi1, Phi2), comment))
# ---------------------------------------------------------------------

def dgp_independent_ar1(T, rng):
    """Two independent AR(1) — Φ_1 diagonal, Φ_2 = 0."""
    eps = rng.standard_normal((T, 2))
    Y = np.zeros((T, 2))
    for t in range(1, T):
        Y[t, 0] = 0.5 * Y[t-1, 0] + eps[t, 0]
        Y[t, 1] = 0.7 * Y[t-1, 1] + eps[t, 1]
    Phi1 = np.diag([0.5, 0.7])
    Phi2 = np.zeros((2, 2))
    return Y, Phi1, Phi2


def dgp_cointegrated(T, rng):
    """y_2 = RW; y_1 = 0.6 y_2 + small AR(1) noise.

    The cointegrating relation: y_1 - 0.6 y_2 ~ stationary.
    rank(M) = 1.
    """
    eps2 = rng.standard_normal(T) * 0.5
    y2 = np.cumsum(eps2)
    u = np.zeros(T)
    eps_u = rng.standard_normal(T) * 0.2
    for t in range(1, T):
        u[t] = 0.5 * u[t-1] + eps_u[t]
    y1 = 0.6 * y2 + u
    Y = np.column_stack([y1, y2])
    # No clean (Φ_1, Φ_2) representation — return None for comparison.
    return Y, None, None


def dgp_triangular_var(T, rng):
    """y_{2,t+1} = 0.7 y_{2,t} + ε; y_{1,t+1} = 0.5 y_{1,t} + 0.3 y_{2,t} + ε."""
    eps = rng.standard_normal((T, 2))
    Y = np.zeros((T, 2))
    for t in range(1, T):
        Y[t, 0] = 0.5 * Y[t-1, 0] + 0.3 * Y[t-1, 1] + eps[t, 0]
        Y[t, 1] = 0.7 * Y[t-1, 1] + eps[t, 1]
    Phi1 = np.array([[0.5, 0.3], [0.0, 0.7]])
    Phi2 = np.zeros((2, 2))
    return Y, Phi1, Phi2


def dgp_mixed_i1_ar1(T, rng):
    """y_1 = RW; y_2 = stationary AR(1)."""
    eps = rng.standard_normal((T, 2))
    Y = np.zeros((T, 2))
    Y[:, 0] = np.cumsum(eps[:, 0])
    for t in range(1, T):
        Y[t, 1] = 0.5 * Y[t-1, 1] + eps[t, 1]
    Phi1 = np.diag([1.0, 0.5])
    Phi2 = np.zeros((2, 2))
    return Y, Phi1, Phi2


def dgp_explosive_mix(T, rng):
    """y_1 explosive (φ=1.05), y_2 stationary AR(1)."""
    eps = rng.standard_normal((T, 2))
    Y = np.zeros((T, 2))
    for t in range(1, T):
        Y[t, 0] = 1.05 * Y[t-1, 0] + eps[t, 0]
        Y[t, 1] = 0.5 * Y[t-1, 1] + eps[t, 1]
    Phi1 = np.diag([1.05, 0.5])
    Phi2 = np.zeros((2, 2))
    return Y, Phi1, Phi2


def dgp_two_indep_rw(T, rng):
    """Two independent random walks (rank(M)=0, no cointegration)."""
    eps = rng.standard_normal((T, 2))
    Y = np.cumsum(eps, axis=0)
    Phi1 = np.eye(2)
    Phi2 = np.zeros((2, 2))
    return Y, Phi1, Phi2


def dgp_full_var1(T, rng):
    """Full VAR(1) with cross-coupling, stationary."""
    Phi1 = np.array([[0.5, 0.2], [0.15, 0.6]])
    eps = rng.standard_normal((T, 2))
    Y = np.zeros((T, 2))
    for t in range(1, T):
        Y[t] = Phi1 @ Y[t-1] + eps[t]
    Phi2 = np.zeros((2, 2))
    return Y, Phi1, Phi2


PANEL = [
    ("M1: independent AR(1)",   dgp_independent_ar1),
    ("M2: cointegrated I(1)",   dgp_cointegrated),
    ("M3: triangular VAR(1)",   dgp_triangular_var),
    ("M4: mixed I(1)+AR(1)",    dgp_mixed_i1_ar1),
    ("M5: explosive + AR(1)",   dgp_explosive_mix),
    ("M6: two indep RW",        dgp_two_indep_rw),
    ("M7: full VAR(1)",         dgp_full_var1),
]


# ---------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------

def matrix_mse(M_hat, M_true):
    if M_true is None:
        return float("nan")
    return float(np.sum((M_hat - M_true) ** 2))


def forecast_one_step_pt(Y, A, B):
    """Forecast y_{T+1} using fitted (A, B)."""
    y_t = Y[-1]
    dy_t = Y[-1] - Y[-2]
    return A @ y_t + B @ dy_t


def forecast_one_step_var(Y, Phi1, Phi2):
    """Forecast using VAR(2) coefficients."""
    return Phi1 @ Y[-1] + Phi2 @ Y[-2]


def run_cell(label, gen_fn):
    rng_master = np.random.default_rng(SEED)
    seeds = rng_master.integers(0, 2**31, size=N_REPS)

    rows = []
    for s in seeds:
        rng = np.random.default_rng(int(s))
        Y, Phi1_true, Phi2_true = gen_fn(T, rng)
        # split for forecasting
        n_train = int(0.85 * T)
        Y_train = Y[:n_train]
        Y_test = Y[n_train:]

        # PT-MV
        A, B = ptmv_level1(Y_train)
        Phi1_pt, Phi2_pt = AB_to_Phi(A, B)
        M = ec_matrix(A, B)
        rank_M, sing_vals = emergent_rank(M, T=Y_train.shape[0], threshold=2.0)

        # VAR(2) baseline
        Phi1_var, Phi2_var = var2_ols(Y_train)

        # forecast errors (recursive 1-step through test)
        history = list(Y_train)
        rmse_pt = []; rmse_var = []
        for y_true in Y_test:
            f_pt = forecast_one_step_pt(np.asarray(history), A, B)
            f_var = forecast_one_step_var(np.asarray(history), Phi1_var, Phi2_var)
            rmse_pt.append(np.linalg.norm(f_pt - y_true) ** 2)
            rmse_var.append(np.linalg.norm(f_var - y_true) ** 2)
            history.append(y_true)
        rmse_pt = float(np.sqrt(np.mean(rmse_pt)))
        rmse_var = float(np.sqrt(np.mean(rmse_var)))

        # Parameter MSE if known
        mse_pt_phi = matrix_mse(Phi1_pt, Phi1_true) + matrix_mse(Phi2_pt, Phi2_true)
        mse_var_phi = matrix_mse(Phi1_var, Phi1_true) + matrix_mse(Phi2_var, Phi2_true)

        rows.append({
            "rmse_pt": rmse_pt, "rmse_var": rmse_var,
            "mse_pt": mse_pt_phi, "mse_var": mse_var_phi,
            "rank_M": rank_M,
        })
    return rows


def main():
    print("=" * 100)
    print(f"ST-Multivariate — PT-MV vs VAR(2) | T={T}, N={N_REPS}")
    print("=" * 100)

    print(f"\n{'cell':>26s} | {'PT-MV RMSE':>11s} {'VAR2 RMSE':>11s} | "
          f"{'PT-MV MSE-Φ':>12s} {'VAR2 MSE-Φ':>12s} | {'rank(M̂)':>8s}")
    print("-" * 95)
    summary = []
    for label, gen in PANEL:
        rows = run_cell(label, gen)
        rmse_pt = np.mean([r["rmse_pt"] for r in rows])
        rmse_var = np.mean([r["rmse_var"] for r in rows])
        mse_pt = np.nanmean([r["mse_pt"] for r in rows])
        mse_var = np.nanmean([r["mse_var"] for r in rows])
        rank_M_mean = np.mean([r["rank_M"] for r in rows])
        # Distribution of rank(M̂) across seeds
        ranks = [r["rank_M"] for r in rows]
        rank_str = f"{int(round(rank_M_mean))} ({min(ranks)}-{max(ranks)})"
        mse_pt_str = f"{mse_pt:>12.4e}" if not np.isnan(mse_pt) else "          —"
        mse_var_str = f"{mse_var:>12.4e}" if not np.isnan(mse_var) else "          —"
        print(f"{label:>26s} | {rmse_pt:>11.4f} {rmse_var:>11.4f} | "
              f"{mse_pt_str} {mse_var_str} | {rank_str:>8s}")
        summary.append({
            "cell": label, "rmse_pt": rmse_pt, "rmse_var": rmse_var,
            "mse_pt": mse_pt, "mse_var": mse_var, "rank_M_mean": rank_M_mean,
        })

    print("\nNotes:")
    print("  rank(M̂) ≈ 0 → no error correction (independent processes)")
    print("  rank(M̂) ≈ 1 → bivariate cointegration with r=1 vector")
    print("  rank(M̂) ≈ 2 → no cointegration / stationary (full rank)")

    out_path = os.path.join(RESULTS_DIR, "st_multivariate_battery.csv")
    import csv as csv_mod
    with open(out_path, "w", newline="") as f:
        w = csv_mod.DictWriter(f, fieldnames=list(summary[0].keys()))
        w.writeheader()
        for row in summary:
            w.writerow(row)
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
