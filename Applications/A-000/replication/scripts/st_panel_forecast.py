"""
ST-Panel — One-step-ahead forecasting on a synthetic regime-mixture
panel (proxy for FRED-MD-style heterogeneous macro/financial data).

The panel consists of 12 series, each generated from a different DGP
spanning the regime spectrum: stationary AR(1)/AR(2), random walk,
near-unit-root, explosive AR, structural-break, AR with trend, GARCH-
like volatility, AR(2) with heavy tails, sine + noise, two chaotic
systems.

For each series, we split into train (first 80%) and test (last 20%),
fit each estimator on train, forecast one step ahead recursively
through the test window, and compute MSFE.

Estimators compared:
  - PT-univ (level-1, regime-agnostic)
  - PT-univ-quadratic (extends to nonlinear via 6-feature basis)
  - OLS-AR(1)
  - OLS-AR(2)
  - Yule-Walker AR(2)
  - Naive (random-walk forecast: ŷ_{t+1} = y_t)
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
)
from action_graded_quadratic import design_quadratic, ptls_quadratic

PROJECT_ROOT = os.path.abspath(os.path.join(HERE, ".."))
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

T = 600
TRAIN_FRAC = 0.8
N_REPS = 20
SEED = 42


# ---------------------------------------------------------------------
# DGPs
# ---------------------------------------------------------------------

def gen_ar1(phi, sigma, T, rng):
    eps = sigma * rng.standard_normal(T)
    y = np.empty(T); y[0] = 0
    for t in range(1, T):
        y[t] = phi * y[t-1] + eps[t]
    return y


def gen_ar2(phi1, phi2, sigma, T, rng):
    eps = sigma * rng.standard_normal(T)
    y = np.zeros(T)
    for t in range(2, T):
        y[t] = phi1 * y[t-1] + phi2 * y[t-2] + eps[t]
    return y


def gen_struct_break(T, rng):
    half = T // 2
    return np.concatenate([
        gen_ar1(0.3, 1.0, half, rng),
        gen_ar1(0.85, 1.0, T - half, rng) + 5.0,
    ])


def gen_garch_like(T, rng):
    """ARCH(1)-style heteroscedastic series."""
    h = np.empty(T); h[0] = 1.0
    eps = np.empty(T)
    for t in range(T):
        eps[t] = np.sqrt(h[t]) * rng.standard_normal()
        if t < T - 1:
            h[t+1] = 0.2 + 0.7 * eps[t]**2
    y = np.empty(T); y[0] = eps[0]
    for t in range(1, T):
        y[t] = 0.5 * y[t-1] + eps[t]
    return y


def gen_ar2_heavy_tail(T, rng):
    z = rng.standard_t(3.0, T)
    y = np.zeros(T)
    for t in range(2, T):
        y[t] = 0.7 * y[t-1] - 0.2 * y[t-2] + z[t]
    return y


def gen_sine_plus_noise(T, rng):
    t = np.arange(T)
    return np.sin(0.25 * t + 0.7) + 0.5 * np.sin(0.6 * t) + 0.1 * rng.standard_normal(T)


def gen_logistic_chaos(T, rng):
    r, y0 = 3.95, 0.4
    y = np.empty(T); y[0] = y0
    for t in range(1, T):
        y[t] = r * y[t-1] * (1 - y[t-1])
    # add tiny observation noise so things are realistic
    return y + 0.005 * rng.standard_normal(T)


def gen_henon(T, rng):
    a, b = 1.4, 0.3
    y = np.empty(T); y[0]=0; y[1]=0
    for t in range(2, T):
        y[t] = 1 - a * y[t-1]**2 + b * y[t-2]
    return y + 0.005 * rng.standard_normal(T)


def gen_trend_plus_ar(T, rng):
    """y_t = α t + AR(1) noise."""
    return 0.01 * np.arange(T) + gen_ar1(0.6, 1.0, T, rng)


PANEL = [
    ("S1: AR(1) stable",     lambda T, r: gen_ar1(0.5, 1.0, T, r)),
    ("S2: AR(1) near unit",  lambda T, r: gen_ar1(0.95, 1.0, T, r)),
    ("S3: Random walk",      lambda T, r: gen_ar1(1.0, 1.0, T, r)),
    ("S4: AR(1) explosive",  lambda T, r: gen_ar1(1.05, 1.0, T, r)),
    ("S5: AR(2) stable",     lambda T, r: gen_ar2(1.4, -0.6, 1.0, T, r)),
    ("S6: AR(2) heavy tail", gen_ar2_heavy_tail),
    ("S7: Struct break",     gen_struct_break),
    ("S8: GARCH-like",       gen_garch_like),
    ("S9: Trend + AR",       gen_trend_plus_ar),
    ("S10: Quasi-periodic",  gen_sine_plus_noise),
    ("S11: Logistic chaos",  gen_logistic_chaos),
    ("S12: Hénon",           gen_henon),
]


# ---------------------------------------------------------------------
# Forecasters
# ---------------------------------------------------------------------

def forecast_pt_univ(y_train, y_test_so_far):
    """Regress y_{t+1} = a y_t + b Δy_t on training, predict next step."""
    a, b = ptls_universal_level1(y_train)
    y_t = y_test_so_far[-1]
    if len(y_test_so_far) >= 2:
        dy_t = y_test_so_far[-1] - y_test_so_far[-2]
    elif len(y_train) >= 1:
        dy_t = y_test_so_far[-1] - y_train[-1]
    else:
        dy_t = 0.0
    return a * y_t + b * dy_t


def forecast_ols_ar1(y_train, y_test_so_far):
    y = y_train
    if len(y) < 3:
        return y_test_so_far[-1]
    Y = y[1:]
    X = y[:-1].reshape(-1, 1)
    coef, *_ = np.linalg.lstsq(X, Y, rcond=None)
    phi = float(coef[0])
    return phi * y_test_so_far[-1]


def forecast_ols_ar2(y_train, y_test_so_far):
    phi1, phi2 = ar2_ols(y_train)
    y_t = y_test_so_far[-1]
    if len(y_test_so_far) >= 2:
        y_tm1 = y_test_so_far[-2]
    elif len(y_train) >= 1:
        y_tm1 = y_train[-1]
    else:
        y_tm1 = 0.0
    return phi1 * y_t + phi2 * y_tm1


def forecast_yw(y_train, y_test_so_far):
    phi1, phi2 = yule_walker_ar2(y_train)
    y_t = y_test_so_far[-1]
    if len(y_test_so_far) >= 2:
        y_tm1 = y_test_so_far[-2]
    elif len(y_train) >= 1:
        y_tm1 = y_train[-1]
    else:
        y_tm1 = 0.0
    return phi1 * y_t + phi2 * y_tm1


def forecast_naive(y_train, y_test_so_far):
    return y_test_so_far[-1]


def forecast_pt_univ_quadratic(y_train, y_test_so_far):
    c = ptls_quadratic(y_train)
    y_t = y_test_so_far[-1]
    if len(y_test_so_far) >= 2:
        dy_t = y_test_so_far[-1] - y_test_so_far[-2]
    else:
        dy_t = y_test_so_far[-1] - y_train[-1] if len(y_train) > 0 else 0.0
    feat = np.array([1.0, y_t, dy_t, y_t**2, dy_t**2, y_t * dy_t])
    return float(c @ feat)


FORECASTERS = [
    ("PT-univ",       forecast_pt_univ),
    ("PT-univ-quad",  forecast_pt_univ_quadratic),
    ("OLS-AR(1)",     forecast_ols_ar1),
    ("OLS-AR(2)",     forecast_ols_ar2),
    ("YW-AR(2)",      forecast_yw),
    ("Naive",         forecast_naive),
]


# ---------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------

def evaluate_panel():
    rng_master = np.random.default_rng(SEED)
    seeds = rng_master.integers(0, 2**31, size=N_REPS)

    # Returns: dict (series_label, forecaster) -> mean(rmse) across reps
    results = {}
    for label, _ in PANEL:
        for fname, _ in FORECASTERS:
            results[(label, fname)] = []

    for s in seeds:
        rng = np.random.default_rng(int(s))
        for series_label, gen in PANEL:
            y = gen(T, rng)
            n_train = int(TRAIN_FRAC * T)
            y_train = y[:n_train]
            y_test = y[n_train:]
            for fname, fn in FORECASTERS:
                preds = []
                # Recursive one-step forecast through test window
                history = list(y_train)
                for i, y_true in enumerate(y_test):
                    pred = fn(np.asarray(history), np.asarray(history))
                    preds.append(pred)
                    history.append(y_true)
                preds = np.asarray(preds)
                rmse = float(np.sqrt(np.mean((preds - y_test) ** 2)))
                results[(series_label, fname)].append(rmse)

    return results


def main():
    print("=" * 100)
    print(f"ST-Panel — One-step-ahead forecasting comparison | "
          f"T={T}, train={int(TRAIN_FRAC*100)}%, N_reps={N_REPS}")
    print("=" * 100)

    t0 = time.time()
    results = evaluate_panel()
    elapsed = time.time() - t0
    print(f"\n[Total: {elapsed:.1f}s]\n")

    # Print one row per series: mean RMSE per forecaster, plus best.
    fnames = [name for name, _ in FORECASTERS]
    header = " | ".join([f"{f:>12s}" for f in fnames])
    print(f"{'series':>22s} | {header}")
    print("-" * (24 + len(fnames) * 15))

    n_wins = {f: 0 for f in fnames}
    for series_label, _ in PANEL:
        cells = []
        means = {}
        for f in fnames:
            errs = np.asarray(results[(series_label, f)])
            means[f] = errs.mean()
            cells.append(means[f])
        best_f = min(fnames, key=lambda f: means[f])
        n_wins[best_f] += 1
        row = []
        for f, m in zip(fnames, cells):
            mark = "*" if f == best_f else " "
            row.append(f"{m:>11.4f}{mark}")
        print(f"{series_label:>22s} | {' | '.join(row)}")

    print("\nWins per forecaster:")
    for f, n in sorted(n_wins.items(), key=lambda kv: -kv[1]):
        print(f"  {f:>14s}: {n}/{len(PANEL)}")

    # Save raw results
    out_path = os.path.join(RESULTS_DIR, "st_panel_forecast.csv")
    with open(out_path, "w") as f:
        f.write("series,forecaster,rmse\n")
        for (sl, fn), errs in results.items():
            for e in errs:
                f.write(f"{sl},{fn},{e}\n")
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
