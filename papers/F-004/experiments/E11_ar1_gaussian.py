"""
E11 — The AR(1) Gaussian trajectory  (depth-1 memory, continuous step)
=======================================================================

The first continuous-step trajectory with memory. The rule is

        X_t  =  φ · X_{t-1}  +  ε_t,     ε_t ~ N(0, 1) iid,

with φ = 0.7. Each step is generated from the previous one plus an
independent Gaussian shock; the shock is exactly the Gaussian step of
E06, and the new ingredient is the term φ·X_{t-1} that couples X_t to
its immediate past.

This is the continuous-step analog of the sticky coin of E05. Memory
is at depth 1 — the next step depends on the most recent one and on
nothing further back. The substrate's operator at depth 1 is a single
number φ; given X_{t-1}, the conditional distribution of X_t is
N(φ X_{t-1}, 1).

What we want to see, by direct numerical comparison:

  (i)   the stationary marginal of X_t is N(0, 1/(1 − φ²)) — the
        Gaussian step from E06, broadened by the memory feedback;

  (ii)  the empirical autocorrelation at lag k is φ^k, decaying
        geometrically (matching the simple AR(1) prediction);

  (iii) the long-run variance of the running sum scales by the
        memory factor: Var(S_n)/n converges to σ²_X · (1+φ)/(1−φ),
        which for φ = 0.7 gives ≈ 11.1, vs the iid prediction of
        σ²_X ≈ 1.96;

  (iv)  the substrate's depth-1 operator is recovered by OLS
        regression of X_t on X_{t-1} — and the residuals are white
        (no remaining autocorrelation), which is the certificate that
        the depth-1 reading is enough for this process.

Outputs
-------
    results/E11_ar1_marginal.csv
    results/E11_ar1_autocorrelation.csv
    results/E11_ar1_variance_scaling.csv
    results/E11_ar1_operator_recovery.csv
    results/E11_ar1_gaussian.md

Run
---
    python3 E11_ar1_gaussian.py
"""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
RESULTS = HERE.parent / "results"
RESULTS.mkdir(exist_ok=True)


PHI = 0.7
SIGMA = 1.0
STATIONARY_VAR = SIGMA ** 2 / (1.0 - PHI ** 2)


# ---------------------------------------------------------------------------
# Generate the trajectory
# ---------------------------------------------------------------------------

def ar1_trajectory(N: int, phi: float, sigma: float,
                   rng: np.random.Generator, burnin: int = 500) -> np.ndarray:
    X = np.empty(N + burnin)
    X[0] = sigma * rng.standard_normal() / np.sqrt(max(1.0 - phi ** 2, 1e-12))
    eps = sigma * rng.standard_normal(N + burnin - 1)
    for t in range(1, N + burnin):
        X[t] = phi * X[t - 1] + eps[t - 1]
    return X[burnin:]


# ---------------------------------------------------------------------------
# Diagnostics
# ---------------------------------------------------------------------------

def autocorrelation_function(X: np.ndarray, max_lag: int) -> np.ndarray:
    Xc = X - X.mean()
    denom = float((Xc * Xc).mean())
    return np.array([
        float((Xc[: len(Xc) - k] * Xc[k:]).mean() / max(denom, 1e-12))
        for k in range(max_lag + 1)
    ])


def variance_scaling_running_sum(X: np.ndarray, horizons: list[int]) -> list[float]:
    """Var(S_n)/n at each horizon, using non-overlapping windows."""
    out = []
    for n in horizons:
        blocks = X[: len(X) // n * n].reshape(-1, n).sum(axis=1)
        out.append(float(blocks.var() / n))
    return out


def fit_ar1(X: np.ndarray) -> dict:
    """OLS regression of X_t on X_{t-1}. Returns φ̂, σ̂², and residuals."""
    y = X[1:]
    x = X[:-1]
    phi_hat = float((x * y).sum() / max((x * x).sum(), 1e-12))
    residuals = y - phi_hat * x
    sigma_hat = float(np.sqrt(residuals.var()))
    return {"phi_hat": phi_hat, "sigma_hat": sigma_hat,
            "residuals": residuals}


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def main():
    rng = np.random.default_rng(0)
    N = 200_000
    print(f"Step 1.  Generate one long AR(1) trajectory of length {N} with "
          f"φ = {PHI}, σ = {SIGMA}.")
    X = ar1_trajectory(N, PHI, SIGMA, rng)
    print(f"  empirical stationary mean  = {X.mean():+.4f}  (expected 0)")
    print(f"  empirical stationary variance = {X.var():.4f}  "
          f"(expected {STATIONARY_VAR:.4f} = σ²/(1−φ²))")

    # --- Marginal -------------------------------------------------------
    bins = 80
    edges = np.linspace(X.min(), X.max(), bins + 1)
    counts, _ = np.histogram(X, bins=edges, density=True)
    centres = 0.5 * (edges[:-1] + edges[1:])
    pdf_analytical = np.exp(-0.5 * centres ** 2 / STATIONARY_VAR) / np.sqrt(
        2.0 * np.pi * STATIONARY_VAR)
    bin_width = float(edges[1] - edges[0])
    l1_marginal = float(np.sum(np.abs(counts - pdf_analytical) * bin_width))
    print(f"  L¹ (empirical marginal vs analytical N(0, σ²_X)) = "
          f"{l1_marginal:.4f}")

    marginal_rows = []
    for i in range(len(centres)):
        marginal_rows.append({
            "bin_centre": float(centres[i]),
            "empirical_density": float(counts[i]),
            "analytical_stationary_density": float(pdf_analytical[i]),
            "abs_diff": float(abs(counts[i] - pdf_analytical[i])),
        })

    # --- Autocorrelation function ---------------------------------------
    print("\nStep 2.  Autocorrelation function at lags 0..10.")
    max_lag = 10
    acf = autocorrelation_function(X, max_lag)
    acf_rows = []
    for k in range(max_lag + 1):
        predicted = PHI ** k
        print(f"  lag k = {k:>2d}   empirical ρ_k = {acf[k]:+.4f}   "
              f"predicted φ^k = {predicted:+.4f}")
        acf_rows.append({
            "lag": int(k),
            "empirical_acf": float(acf[k]),
            "predicted_phi_power_k": float(predicted),
        })

    # --- Variance scaling of running sum --------------------------------
    print("\nStep 3.  Var(S_n)/n at several horizons, comparing iid prediction"
          " (σ²_X) vs memory-corrected prediction (σ²_X · (1+φ)/(1−φ)).")
    horizons = [1, 5, 20, 100, 1000]
    var_per_step = variance_scaling_running_sum(X, horizons)
    iid_pred = STATIONARY_VAR
    memory_pred = STATIONARY_VAR * (1.0 + PHI) / (1.0 - PHI)
    variance_rows = []
    for n, v in zip(horizons, var_per_step):
        print(f"  n = {n:>4d}   Var(S_n)/n (emp) = {v:>8.4f}   "
              f"iid prediction = {iid_pred:.4f}   "
              f"memory prediction = {memory_pred:.4f}")
        variance_rows.append({
            "horizon": n,
            "var_per_step_empirical": v,
            "iid_prediction": iid_pred,
            "memory_prediction_long_run": memory_pred,
        })

    # --- Operator recovery at depth 1 -----------------------------------
    print("\nStep 4.  Substrate's depth-1 operator: OLS of X_t on X_{t-1}.")
    fit = fit_ar1(X)
    print(f"  φ̂ (empirical)     = {fit['phi_hat']:+.4f}   (planted {PHI})")
    print(f"  σ̂ (residuals)     = {fit['sigma_hat']:.4f}   (planted {SIGMA})")
    resid_acf = autocorrelation_function(fit["residuals"], max_lag=5)
    print(f"  ACF of residuals (lags 1..5): "
          f"{[round(r, 4) for r in resid_acf[1:]]}")
    print(f"    (expected ≈ 0 at all lags if the depth-1 fit captured "
          f"all the structure)")

    op_rows = [
        {"parameter": "phi", "empirical": fit["phi_hat"], "planted": PHI},
        {"parameter": "sigma_innovation", "empirical": fit["sigma_hat"],
         "planted": SIGMA},
    ]
    for k, r in enumerate(resid_acf[1:], start=1):
        op_rows.append({"parameter": f"residual_acf_lag_{k}",
                        "empirical": float(r),
                        "planted": 0.0})

    # --- CSVs ------------------------------------------------------------
    with (RESULTS / "E11_ar1_marginal.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["bin_centre",
                                          "empirical_density",
                                          "analytical_stationary_density",
                                          "abs_diff"])
        w.writeheader()
        for r in marginal_rows:
            w.writerow(r)

    with (RESULTS / "E11_ar1_autocorrelation.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["lag", "empirical_acf",
                                          "predicted_phi_power_k"])
        w.writeheader()
        for r in acf_rows:
            w.writerow(r)

    with (RESULTS / "E11_ar1_variance_scaling.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["horizon",
                                          "var_per_step_empirical",
                                          "iid_prediction",
                                          "memory_prediction_long_run"])
        w.writeheader()
        for r in variance_rows:
            w.writerow(r)

    with (RESULTS / "E11_ar1_operator_recovery.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["parameter", "empirical", "planted"])
        w.writeheader()
        for r in op_rows:
            w.writerow(r)

    # --- Readout ---------------------------------------------------------
    md = [
        f"# E11 — The AR(1) Gaussian trajectory  (φ = {PHI})",
        "",
        f"**Rule:** X_t = φ X_{{t-1}} + ε_t, with ε_t ~ N(0, 1) iid and "
        f"φ = {PHI}.  **Trajectory length:** N = {N}.  "
        "**Stationary marginal:** N(0, σ²/(1 − φ²)).",
        "",
        "## What is new in this experiment",
        "",
        "First trajectory with continuous-step memory. The Gaussian step "
        "of E06 has been carried over unchanged — the innovation ε_t is "
        "the same N(0, 1) draw — and a single new ingredient has been "
        "added: the term φ·X_{t-1} that couples X_t to its immediate past. "
        "This is the continuous-step analog of the sticky coin (E05), with "
        "the same depth of memory but a continuous state instead of a "
        "two-valued one.",
        "",
        "## The stationary marginal",
        "",
        "Empirical first two moments of the trajectory:",
        "",
        "| quantity | empirical | expected |",
        "| :--- | ---: | ---: |",
        f"| mean             | {X.mean():+.4f} | 0 |",
        f"| variance         | {X.var():.4f} | {STATIONARY_VAR:.4f} = σ²/(1−φ²) |",
        f"| L¹ (hist vs density) | {l1_marginal:.4f} | sampling noise |",
        "",
        "The stationary marginal is exactly the Gaussian of E06, but with "
        f"variance broadened by the memory feedback to σ²/(1 − φ²) ≈ "
        f"{STATIONARY_VAR:.2f} instead of 1. Same shape, wider.",
        "",
        "## The autocorrelation function decays geometrically",
        "",
        "Empirical ρ_k at lags 0..10 vs the prediction ρ_k = φ^k:",
        "",
        "| lag k | empirical ρ_k | predicted φ^k |",
        "| ---: | ---: | ---: |",
    ]
    for r in acf_rows:
        md.append(f"| {r['lag']} | {r['empirical_acf']:+.4f} | "
                  f"{r['predicted_phi_power_k']:+.4f} |")

    md += [
        "",
        "Match within sampling noise at every lag. The correlation between "
        "X_t and X_{t+k} dies as φ^k = 0.7^k — at lag 10 it is already "
        "below 3%. The memory is short-range in absolute terms (a few "
        "steps suffice to forget) but it is still genuinely depth-1: at "
        "each step the next state is conditioned on the *previous* state, "
        "with all earlier states contributing only through that one.",
        "",
        "## The variance of the running sum scales by a memory factor",
        "",
        "If the steps were independent (no memory), the variance of S_n "
        f"would simply be n · σ²_X = n · {STATIONARY_VAR:.2f}. With memory, "
        "the long-run variance per step is",
        "",
        f"    σ²_X · (1 + φ) / (1 − φ)  =  "
        f"{STATIONARY_VAR:.2f} · "
        f"{(1+PHI)/(1-PHI):.2f}  ≈  {memory_pred:.2f},",
        "",
        f"a factor of {(1+PHI)/(1-PHI):.2f} larger than the iid value. "
        "Empirically:",
        "",
        "| horizon n | Var(S_n)/n (empirical) | iid prediction | memory-corrected prediction (long-run) |",
        "| ---: | ---: | ---: | ---: |",
    ]
    for r in variance_rows:
        md.append(f"| {r['horizon']} | "
                  f"{r['var_per_step_empirical']:.4f} | "
                  f"{r['iid_prediction']:.4f} | "
                  f"{r['memory_prediction_long_run']:.4f} |")

    md += [
        "",
        "Var(S_n)/n approaches the memory-corrected prediction at large n. "
        "At horizon 1 it equals σ²_X exactly (one step, no accumulation of "
        "memory). At horizons 100 – 1000 it sits at the long-run value. "
        "The persistence inflates the variance by ≈ 5.7× compared to the "
        "iid expectation. Same statement as in E05 — same machinery, "
        "different state space.",
        "",
        "## The substrate's depth-1 operator is recovered cleanly",
        "",
        "Fitting the depth-1 operator by OLS regression of X_t on X_{t-1}:",
        "",
        "| parameter | empirical | planted |",
        "| :--- | ---: | ---: |",
        f"| φ                                       | {fit['phi_hat']:+.4f} | {PHI:+.4f} |",
        f"| innovation σ (residual standard dev)   | {fit['sigma_hat']:.4f} | {SIGMA:.4f} |",
        "",
        "Residual autocorrelation function (after subtracting the depth-1 "
        "fit) at lags 1 – 5:",
        "",
        "| lag | residual ACF |",
        "| ---: | ---: |",
    ]
    for k, r in enumerate(resid_acf[1:], start=1):
        md.append(f"| {k} | {r:+.4f} |")

    md += [
        "",
        "All residual autocorrelations are at sampling-noise level (~ "
        "0.01). The depth-1 operator captures all the dependence in the "
        "trajectory; nothing is left for a depth-2 or deeper reading to "
        "find. **This is the certificate that the operator at the right "
        "depth captures the process fully** — when the trajectory is AR(1), "
        "the depth-1 fit's residuals are white.",
        "",
        "## What this experiment shows",
        "",
        "The dependence-axis tools we built for the discrete sticky coin "
        "in E05 transfer to continuous-state trajectories without change. "
        "The autocorrelation function decays geometrically as φ^k; the "
        "variance of the running sum is inflated by exactly the same "
        "(1+ρ)/(1−ρ) factor with ρ = φ; the depth-1 operator (a single "
        "number φ) is recovered by OLS and the residuals are white. The "
        "substrate's depth-1 reading is, for this process, the entire "
        "story.",
        "",
        "The natural next question is what happens when the dependence is "
        "deeper than depth 1 — when the next step depends on more than "
        "just the immediately previous one. That is E12 (AR(2)), where "
        "the depth-1 reading is no longer enough and the residuals expose "
        "the missing structure.",
        "",
    ]
    (RESULTS / "E11_ar1_gaussian.md").write_text("\n".join(md))
    print(f"\nWrote {RESULTS / 'E11_ar1_gaussian.md'}")


if __name__ == "__main__":
    main()
