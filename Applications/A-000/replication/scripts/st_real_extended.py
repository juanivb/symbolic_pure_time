"""
ST-Real-Extended — Extended battery of real-data tests.

Adds to the previous st_real_cointegration.py:
  R7: Consumption-income (PCECC96 vs DPIC96, Hall 1978 hypothesis)
  R8: M1 vs M2 monetary aggregates (FRED monthly)
  R9: FX triangulation (JPY/USD, EUR/USD, GBP/USD daily)
  R10: Argentina USD/ARS in explosive regime (2017-2025)
  R11: Nominal vs real interest rate differential
  R12: GDP vs Industrial Production (real activity)

Plus the Lorenz multivariate-quadratic recovery from st_lorenz_recovery.py
in summary form.
"""
from __future__ import annotations

import os
import sys
import csv
import time

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from action_graded_multivariate import (
    ptmv_level1, ec_matrix, emergent_rank, var2_ols,
)

DATA_DIR = "/sessions/busy-bold-brown/mnt/data"
PROJECT_ROOT = os.path.abspath(os.path.join(HERE, ".."))
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")
os.makedirs(RESULTS_DIR, exist_ok=True)


def adf_simple(u, max_lag=4):
    u = np.asarray(u, dtype=float).ravel()
    n = len(u)
    if n < max_lag + 5:
        return float("nan")
    du = np.diff(u)
    u_lag = u[max_lag:-1]
    target = du[max_lag:]
    lagged = [du[max_lag - i: -i] for i in range(1, max_lag + 1)]
    X = np.column_stack([u_lag] + lagged)
    coef, *_ = np.linalg.lstsq(X, target, rcond=None)
    fitted = X @ coef
    resid = target - fitted
    sigma2 = float((resid @ resid) / (X.shape[0] - X.shape[1]))
    XtX_inv = np.linalg.pinv(X.T @ X)
    se = float(np.sqrt(sigma2 * XtX_inv[0, 0]))
    return float(coef[0] / se)


def rolling_forecast(Y, fit_fn, train_frac=0.85):
    Y = np.asarray(Y)
    T, d = Y.shape
    n_train = int(train_frac * T)
    fits = fit_fn(Y[:n_train])
    history = list(Y[:n_train])
    errors = []
    for y_true in Y[n_train:]:
        h = np.asarray(history)
        if len(fits) == 2 and isinstance(fits[0], np.ndarray) and fits[0].ndim == 2:
            M1, M2 = fits
            # heuristic: PT-MV uses (y_t, Δy_t); VAR uses (y_{t-1}, y_{t-2})
            # signature distinguished by caller; here we assume PT-MV
            y_t = h[-1]; dy_t = h[-1] - h[-2]
            pred = M1 @ y_t + M2 @ dy_t
        errors.append(float(np.sum((pred - y_true) ** 2)))
        history.append(y_true)
    return float(np.sqrt(np.mean(errors)))


def rolling_forecast_var(Y, train_frac=0.85):
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


def analyse(label, Y, columns, do_eg=True):
    T, d = Y.shape
    print(f"\n{'='*72}")
    print(f"  {label}  |  vars = {columns}  |  T = {T}, d = {d}")
    print(f"{'='*72}")
    A, B = ptmv_level1(Y)
    M = ec_matrix(A, B)
    rank, sing = emergent_rank(M, T=T, threshold=2.0)
    print(f"\n  PT-MV rank diagnostic (threshold σ√T > 2):")
    for i, s in enumerate(sing):
        sR = s * np.sqrt(T)
        flag = "  ← effective" if sR > 2.0 else "  ← spurious"
        print(f"    σ_{i+1} = {s:>10.4e}   σ√T = {sR:>9.3f}{flag}")
    print(f"  → emergent r̂ = {rank}")

    eg_t = float("nan")
    if d == 2 and do_eg:
        y1 = Y[:, 0]; y2 = Y[:, 1]
        slope = float((y1 - y1.mean()) @ (y2 - y2.mean())
                      / ((y2 - y2.mean()) @ (y2 - y2.mean()) + 1e-12))
        intercept = float(y1.mean() - slope * y2.mean())
        u = y1 - slope * y2 - intercept
        eg_t = adf_simple(u)
        eg_flag = "DETECTED" if eg_t < -2.86 else "not detected"
        print(f"\n  Engle-Granger ADF: t = {eg_t:+.3f}, "
              f"slope = {slope:+.4f}  → {eg_flag}")

    rmse_pt = rolling_forecast(Y, ptmv_level1)
    rmse_var = rolling_forecast_var(Y)
    print(f"\n  Rolling 1-step RMSE: PT-MV = {rmse_pt:.6f}   "
          f"VAR(2) = {rmse_var:.6f}   ratio = {rmse_pt/rmse_var:.4f}")
    return {"label": label, "T": T, "d": d, "rank": rank,
            "eg_t": eg_t, "rmse_pt": rmse_pt, "rmse_var": rmse_var}


def main():
    print("=" * 72)
    print(" ST-Real-Extended — additional macro/financial cointegration tests")
    print("=" * 72)
    summary = []

    # R7: Consumption-Income (Hall 1978)
    qf = pd.read_csv(os.path.join(DATA_DIR, "financial/bronze/fred_quarterly_2026_02.csv"),
                     skiprows=[1, 2])
    qf["sasdate"] = pd.to_datetime(qf["sasdate"], format="%m/%d/%Y", errors="coerce")
    pair = qf[["sasdate", "PCECC96", "DPIC96"]].dropna()
    Y = np.log(pair[["PCECC96", "DPIC96"]].values)
    summary.append(analyse("R7: Consumption-Income (Hall 1978)",
                              Y, ["log_PCECC96", "log_DPIC96"]))

    # R8: M1 vs M2 monetary aggregates
    mf = pd.read_csv(os.path.join(DATA_DIR, "financial/bronze/fred_monthly_2026_02.csv"),
                     skiprows=[1])
    mf["sasdate"] = pd.to_datetime(mf["sasdate"], format="%m/%d/%Y", errors="coerce")
    pair = mf[["sasdate", "M1SL", "M2SL"]].dropna()
    Y = np.log(pair[["M1SL", "M2SL"]].values)
    summary.append(analyse("R8: M1 vs M2 (monetary aggregates)",
                              Y, ["log_M1", "log_M2"]))

    # R9: FX triangulation — three currencies vs USD
    jp = pd.read_csv(os.path.join(DATA_DIR, "financial/bronze/fred_dexjpus_extended.csv"))
    eu = pd.read_csv(os.path.join(DATA_DIR, "financial/bronze/fred_dexuseu_extended.csv"))
    uk = pd.read_csv(os.path.join(DATA_DIR, "financial/bronze/fred_dexusuk.csv"))
    for d in (jp, eu, uk):
        d["observation_date"] = pd.to_datetime(d["observation_date"])
    fx = jp.merge(eu, on="observation_date").merge(uk, on="observation_date").dropna()
    # All three pairs in log-FX
    Y = np.log(fx[["DEXJPUS", "DEXUSEU", "DEXUSUK"]].values)
    summary.append(analyse("R9: FX triangulation JPY/EUR/GBP",
                              Y, ["log_JPYUSD", "log_USDEUR", "log_USDGBP"], do_eg=False))

    # R10: Argentina USD/ARS in explosive regime
    arg = pd.read_csv(os.path.join(DATA_DIR, "financial/bronze/argentina_macro_monthly.csv"))
    arg["date"] = pd.to_datetime(arg["date"])
    pair = arg[["date", "usdars", "inflation_monthly_pct"]].dropna()
    Y = np.column_stack([np.log(pair["usdars"].values), pair["inflation_monthly_pct"].values])
    summary.append(analyse("R10: Argentina USD/ARS vs inflation (explosive)",
                              Y, ["log_USDARS", "inflation_pct"]))

    # R11: GDP vs INDPRO (real activity)
    pair = qf[["sasdate", "GDPC1", "INDPRO"]].dropna()
    Y = np.log(pair[["GDPC1", "INDPRO"]].values)
    summary.append(analyse("R11: GDP vs Industrial Production",
                              Y, ["log_GDP", "log_INDPRO"]))

    # R12: Nominal Treasury vs Real (10Y vs 10Y - Inflation)
    pair = mf[["sasdate", "GS10", "CPIAUCSL"]].dropna().reset_index(drop=True)
    pair["dlogCPI_ann"] = 1200.0 * np.concatenate([[np.nan],
                                                    np.diff(np.log(pair["CPIAUCSL"].values))])
    pair = pair.dropna()
    pair["real10y"] = pair["GS10"].values - pair["dlogCPI_ann"].values
    Y = pair[["GS10", "real10y"]].values
    summary.append(analyse("R12: 10Y nominal vs 10Y real ex-post",
                              Y, ["GS10", "real10y"]))

    # Save summary
    out = os.path.join(RESULTS_DIR, "st_real_extended.csv")
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(summary[0].keys()))
        w.writeheader()
        for row in summary:
            w.writerow(row)
    print(f"\nWrote {out}")

    # Quick print summary table
    print("\n" + "=" * 72)
    print(" Summary table")
    print("=" * 72)
    print(f"{'cell':>42s} | {'T':>5s} | {'d':>2s} | {'r̂':>3s} | "
          f"{'EG t':>8s} | {'RMSE-ratio (PT/VAR)':>20s}")
    for s in summary:
        eg_str = f"{s['eg_t']:+.2f}" if not np.isnan(s['eg_t']) else "  —"
        ratio = s['rmse_pt'] / max(s['rmse_var'], 1e-12)
        print(f"{s['label']:>42s} | {s['T']:>5d} | {s['d']:>2d} | "
              f"{s['rank']:>3d} | {eg_str:>8s} | {ratio:>20.4f}")


if __name__ == "__main__":
    main()
