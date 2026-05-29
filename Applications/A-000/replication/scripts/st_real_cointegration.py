"""
ST-Real-Cointegration — PT-MV vs VAR(2) vs Engle-Granger on real data.

Three classical cointegration test pairs from FRED + Yahoo:

  R1: Term structure (TB3MS, GS10)        — interest rate spread
  R2: Yield curve (GS1, GS5, GS10)        — three-tenor yield curve
  R3: Stock pair (XOM, CVX)               — sector pair-trading
  R4: FX (EXUSEU, EXUSUK)                 — major currency pair

For each pair/tuple:
  1. Estimate PT-MV (A, B), compute Π̂ = A + B - I and emergent rank.
  2. Compare with VAR(2) baseline and Engle-Granger 2-step.
  3. Report rank diagnostic + 1-step forecast RMSE on rolling window.
"""
from __future__ import annotations

import os
import sys
import time
import csv

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from action_graded_multivariate import (
    ptmv_level1, AB_to_Phi, ec_matrix, emergent_rank,
    var2_ols, vecm_engle_granger_2step,
)

DATA_DIR = "/sessions/busy-bold-brown/mnt/data"
PROJECT_ROOT = os.path.abspath(os.path.join(HERE, ".."))
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")
os.makedirs(RESULTS_DIR, exist_ok=True)


# ---------------------------------------------------------------------
# Data loaders
# ---------------------------------------------------------------------

def load_fred_monthly_pairs():
    """Load FRED monthly panel and extract canonical cointegration pairs."""
    path = os.path.join(DATA_DIR, "financial/bronze/fred_monthly_2026_02.csv")
    # First row is column names, second row is "Transform:" with codes.
    df = pd.read_csv(path, skiprows=[1])
    df.columns = [c.strip() for c in df.columns]
    # Parse date column
    df["sasdate"] = pd.to_datetime(df["sasdate"], format="%m/%d/%Y", errors="coerce")
    df = df.dropna(subset=["sasdate"]).sort_values("sasdate").reset_index(drop=True)
    return df


def load_stock_pairs():
    path = os.path.join(DATA_DIR, "financial/bronze/pairs_data_fred.csv")
    df = pd.read_csv(path)
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df = df.dropna(subset=["Date"]).sort_values("Date").reset_index(drop=True)
    return df


# ---------------------------------------------------------------------
# Engle-Granger ADF test (no scipy → simple ADF on residual)
# ---------------------------------------------------------------------

def adf_simple(u: np.ndarray, max_lag: int = 4) -> tuple[float, float]:
    """Crude ADF: regress Δu on u_{t-1} + lagged Δu_{t-i}.
    Returns (t-stat, coefficient). t-stat much more negative than -2.86
    indicates rejection of unit root → cointegration."""
    u = np.asarray(u, dtype=float).ravel()
    n = len(u)
    if n < max_lag + 5:
        return float("nan"), float("nan")
    du = np.diff(u)
    u_lag = u[max_lag:-1]                                # u_{t-1}
    target = du[max_lag:]                                # Δu_t
    # Lagged differences
    lagged = []
    for i in range(1, max_lag + 1):
        lagged.append(du[max_lag - i: -i])
    X = np.column_stack([u_lag] + lagged)
    coef, res, rank, *_ = np.linalg.lstsq(X, target, rcond=None)
    fitted = X @ coef
    resid = target - fitted
    sigma2 = float((resid @ resid) / (X.shape[0] - X.shape[1]))
    XtX_inv = np.linalg.pinv(X.T @ X)
    se_coef0 = float(np.sqrt(sigma2 * XtX_inv[0, 0]))
    t_stat = float(coef[0] / se_coef0)
    return t_stat, float(coef[0])


# ---------------------------------------------------------------------
# Rolling-window forecast eval
# ---------------------------------------------------------------------

def rolling_forecast_pt(Y, train_frac=0.8):
    """PT-MV one-step-ahead rolling forecast: y_{t+1} = A y_t + B Δy_t."""
    Y = np.asarray(Y)
    T, d = Y.shape
    n_train = int(train_frac * T)
    A, B = ptmv_level1(Y[:n_train])
    history = list(Y[:n_train])
    errors = []
    for y_true in Y[n_train:]:
        h = np.asarray(history)
        y_t = h[-1]
        dy_t = h[-1] - h[-2]
        pred = A @ y_t + B @ dy_t
        errors.append(float(np.sum((pred - y_true) ** 2)))
        history.append(y_true)
    return float(np.sqrt(np.mean(errors)))


def rolling_forecast_var(Y, train_frac=0.8):
    """VAR(2) one-step-ahead rolling forecast: y_{t+1} = Φ_1 y_t + Φ_2 y_{t-1}."""
    Y = np.asarray(Y)
    T, d = Y.shape
    n_train = int(train_frac * T)
    Phi1, Phi2 = var2_ols(Y[:n_train])
    history = list(Y[:n_train])
    errors = []
    for y_true in Y[n_train:]:
        h = np.asarray(history)
        pred = Phi1 @ h[-1] + Phi2 @ h[-2]
        errors.append(float(np.sum((pred - y_true) ** 2)))
        history.append(y_true)
    return float(np.sqrt(np.mean(errors)))


# ---------------------------------------------------------------------
# Pair analysis
# ---------------------------------------------------------------------

def analyse_pair(label, Y, columns):
    """Run PT-MV, VAR(2), Engle-Granger ADF on a (T, d) panel."""
    T, d = Y.shape
    print(f"\n{'='*72}")
    print(f"  {label}  |  vars = {columns}  |  T = {T}, d = {d}")
    print(f"{'='*72}")

    A, B = ptmv_level1(Y)
    Phi1_pt, Phi2_pt = AB_to_Phi(A, B)
    M = ec_matrix(A, B)
    rank_M, sing = emergent_rank(M, T=T, threshold=2.0)
    print(f"\n PT-MV emergent rank diagnostic:")
    print(f"   singular values of M̂ = Â + B̂ - I:")
    for i, s in enumerate(sing):
        s_root = s * np.sqrt(T)
        flag = "  ← effective" if s_root > 2.0 else "  ← spurious"
        print(f"     σ_{i+1} = {s:>10.4e}   σ√T = {s_root:>9.3f}{flag}")
    print(f"   → emergent rank r̂ = {rank_M}")

    # Engle-Granger 2-step (only meaningful for d=2)
    if d == 2:
        # regress y_1 on y_2
        y1 = Y[:, 0]; y2 = Y[:, 1]
        slope = float((y1 - y1.mean()) @ (y2 - y2.mean())
                      / ((y2 - y2.mean()) @ (y2 - y2.mean())))
        intercept = float(y1.mean() - slope * y2.mean())
        u = y1 - slope * y2 - intercept
        t_stat, coef = adf_simple(u)
        eg_cointegrated = t_stat < -2.86
        print(f"\n Engle-Granger 2-step (d=2):")
        print(f"   y_1 = {intercept:+.4f} + {slope:+.4f} y_2 + u")
        print(f"   ADF on u: t-stat = {t_stat:+.3f}, coef = {coef:+.4f}")
        print(f"   → Cointegration {'DETECTED' if eg_cointegrated else 'NOT detected'} "
              f"at 5% (cv = -2.86)")

    # Forecasting
    rmse_pt = rolling_forecast_pt(Y)
    rmse_var = rolling_forecast_var(Y)
    print(f"\n Rolling 1-step forecast RMSE (train 80%):")
    print(f"   PT-MV  = {rmse_pt:.6f}")
    print(f"   VAR(2) = {rmse_var:.6f}")

    return {
        "label": label,
        "vars": "_".join(columns),
        "T": T,
        "rank_emergent": rank_M,
        "sigmas": list(sing),
        "rmse_pt": rmse_pt,
        "rmse_var": rmse_var,
    }


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main():
    print("=" * 72)
    print(" ST-Real-Cointegration — PT-MV on FRED + Yahoo data")
    print("=" * 72)

    summary = []

    # R1: Term structure (TB3MS, GS10)
    fred = load_fred_monthly_pairs()
    pair_term = fred[["sasdate", "TB3MS", "GS10"]].dropna()
    Y = pair_term[["TB3MS", "GS10"]].values
    summary.append(analyse_pair("R1: Term structure", Y, ["TB3MS", "GS10"]))

    # R2: Yield curve (GS1, GS5, GS10) — d=3
    pair_curve = fred[["sasdate", "GS1", "GS5", "GS10"]].dropna()
    Y = pair_curve[["GS1", "GS5", "GS10"]].values
    summary.append(analyse_pair("R2: Yield curve (3 tenors)", Y, ["GS1", "GS5", "GS10"]))

    # R3: Real-nominal interest rate (TB3MS vs CPI inflation)
    fr = fred[["sasdate", "TB3MS", "CPIAUCSL"]].dropna()
    cpi_inflation = 1200.0 * np.diff(np.log(fr["CPIAUCSL"].values))   # annualised
    Y = np.column_stack([fr["TB3MS"].values[1:], cpi_inflation])
    summary.append(analyse_pair("R3: Nominal rate vs inflation",
                                  Y, ["TB3MS", "CPI_infl_pct"]))

    # R4: Stock pair (XOM, CVX) — daily
    stocks = load_stock_pairs()
    pair_stock = stocks[["Date", "XOM", "CVX"]].dropna()
    Y = pair_stock[["XOM", "CVX"]].values
    # log prices
    Y_log = np.log(Y)
    summary.append(analyse_pair("R4: XOM vs CVX (log-price)",
                                  Y_log, ["log_XOM", "log_CVX"]))

    # R5: Stock pair GS vs MS (financial sector)
    pair_gsms = stocks[["Date", "GS", "MS"]].dropna()
    Y = np.log(pair_gsms[["GS", "MS"]].values)
    summary.append(analyse_pair("R5: GS vs MS (log-price)",
                                  Y, ["log_GS", "log_MS"]))

    # R6: Gold pair GLD vs GDX
    pair_gold = stocks[["Date", "GLD", "GDX"]].dropna()
    Y = np.log(pair_gold[["GLD", "GDX"]].values)
    summary.append(analyse_pair("R6: GLD vs GDX (log-price)",
                                  Y, ["log_GLD", "log_GDX"]))

    # Save summary
    out_path = os.path.join(RESULTS_DIR, "st_real_cointegration.csv")
    with open(out_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["label", "vars", "T", "rank_emergent",
                    "sigmas", "rmse_pt", "rmse_var"])
        for s in summary:
            w.writerow([s["label"], s["vars"], s["T"], s["rank_emergent"],
                        ";".join(f"{x:.4e}" for x in s["sigmas"]),
                        s["rmse_pt"], s["rmse_var"]])
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
