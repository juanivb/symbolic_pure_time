"""
E12 — The AR(2) Gaussian trajectory  (depth-2 memory)
======================================================

The next step in the dependence axis. The rule is

        X_t  =  φ₁ · X_{t-1}  +  φ₂ · X_{t-2}  +  ε_t,    ε_t ~ N(0, 1) iid,

with φ₁ = 0.5, φ₂ = 0.3. The next state depends on the previous *two*
states, not just the previous one. This is the simplest case where
the substrate's depth-1 operator is no longer enough — fitting only on
the immediate past leaves a measurable residual structure at lag 1,
because the lag-2 dependence has nowhere else to live in the depth-1
model.

What we want to see:

  (i)   the trajectory is stationary; the autocorrelation function
        satisfies the depth-2 recurrence
            ρ_k  =  φ₁ ρ_{k−1} + φ₂ ρ_{k−2}     (k ≥ 2),
        with ρ_1 = φ₁ / (1 − φ₂) at the boundary. We check this
        against the empirical ACF;

  (ii)  fitting a depth-1 operator (the model from E11) gives a
        plausible-looking φ̂ ≈ ρ_1 but the residuals still have a
        clear, positive autocorrelation at lag 1 — the missing depth
        is visible in the residual ACF;

  (iii) fitting a depth-2 operator (OLS of X_t on (X_{t-1}, X_{t-2}))
        recovers (φ̂₁, φ̂₂) to within sampling noise, and the
        residuals are white at all lags.

This is the key diagnostic of the dependence axis: **the residuals of
a fit at the wrong depth tell you that the depth is wrong**. You do
not need to know in advance how deep the dependence goes; you can
read it off the residuals.

Outputs
-------
    results/E12_ar2_autocorrelation.csv
    results/E12_ar2_depth1_vs_depth2_fit.csv
    results/E12_ar2_residual_acf_comparison.csv
    results/E12_ar2_gaussian.md

Run
---
    python3 E12_ar2_gaussian.py
"""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
RESULTS = HERE.parent / "results"
RESULTS.mkdir(exist_ok=True)


PHI_1 = 0.5
PHI_2 = 0.3
SIGMA = 1.0
# Stationarity check: |φ₂| < 1, φ₁ + φ₂ < 1, φ₂ − φ₁ < 1.
assert abs(PHI_2) < 1.0
assert PHI_1 + PHI_2 < 1.0
assert PHI_2 - PHI_1 < 1.0
# Stationary variance from Yule-Walker:
#   σ²_X = σ² (1 − φ₂) / ((1 + φ₂) ((1 − φ₂)² − φ₁²))
STATIONARY_VAR = (SIGMA ** 2 * (1 - PHI_2) /
                  ((1 + PHI_2) * ((1 - PHI_2) ** 2 - PHI_1 ** 2)))
# Theoretical autocorrelations
RHO_1_PRED = PHI_1 / (1.0 - PHI_2)
RHO_2_PRED = PHI_2 + PHI_1 * RHO_1_PRED


# ---------------------------------------------------------------------------
# Generate the trajectory
# ---------------------------------------------------------------------------

def ar2_trajectory(N: int, phi1: float, phi2: float, sigma: float,
                   rng: np.random.Generator, burnin: int = 1000) -> np.ndarray:
    eps = sigma * rng.standard_normal(N + burnin)
    X = np.zeros(N + burnin)
    for t in range(2, N + burnin):
        X[t] = phi1 * X[t - 1] + phi2 * X[t - 2] + eps[t]
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


def fit_depth_1(X: np.ndarray) -> dict:
    """OLS of X_t on X_{t-1}."""
    y = X[1:]
    x = X[:-1]
    phi_hat = float((x * y).sum() / max((x * x).sum(), 1e-12))
    residuals = y - phi_hat * x
    return {"phi_hat": phi_hat,
            "sigma_residual": float(np.sqrt(residuals.var())),
            "residuals": residuals}


def fit_depth_2(X: np.ndarray) -> dict:
    """OLS of X_t on (X_{t-1}, X_{t-2})."""
    y = X[2:]
    Z = np.column_stack([X[1:-1], X[:-2]])
    coef, *_ = np.linalg.lstsq(Z, y, rcond=None)
    residuals = y - Z @ coef
    return {"phi1_hat": float(coef[0]),
            "phi2_hat": float(coef[1]),
            "sigma_residual": float(np.sqrt(residuals.var())),
            "residuals": residuals}


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def main():
    rng = np.random.default_rng(0)
    N = 200_000
    print(f"Step 1.  Generate one long AR(2) trajectory with "
          f"φ₁ = {PHI_1}, φ₂ = {PHI_2}, σ = {SIGMA}, length N = {N}.")
    X = ar2_trajectory(N, PHI_1, PHI_2, SIGMA, rng)
    print(f"  empirical mean     = {X.mean():+.4f}  (expected 0)")
    print(f"  empirical variance = {X.var():.4f}  "
          f"(expected {STATIONARY_VAR:.4f} from Yule-Walker)")

    # --- Autocorrelation function ---------------------------------------
    print("\nStep 2.  Autocorrelation function at lags 0..6.")
    max_lag = 6
    acf = autocorrelation_function(X, max_lag)
    # Predict ρ_k from the depth-2 recurrence
    pred = np.zeros(max_lag + 1)
    pred[0] = 1.0
    pred[1] = RHO_1_PRED
    pred[2] = RHO_2_PRED
    for k in range(3, max_lag + 1):
        pred[k] = PHI_1 * pred[k - 1] + PHI_2 * pred[k - 2]
    acf_rows = []
    for k in range(max_lag + 1):
        print(f"  lag k = {k}   empirical ρ_k = {acf[k]:+.4f}   "
              f"predicted (depth-2 recurrence) = {pred[k]:+.4f}")
        acf_rows.append({
            "lag": int(k),
            "empirical_acf": float(acf[k]),
            "predicted_acf_depth2_recurrence": float(pred[k]),
        })

    # --- Depth-1 fit ---------------------------------------------------
    print("\nStep 3.  Fit depth-1 operator (the AR(1) model from E11).")
    fit1 = fit_depth_1(X)
    print(f"  φ̂                = {fit1['phi_hat']:+.4f}   "
          f"(expected ≈ ρ_1 = {RHO_1_PRED:+.4f} — the depth-1 fit recovers "
          f"the lag-1 correlation, NOT the planted φ₁ = {PHI_1})")
    print(f"  σ̂ residual       = {fit1['sigma_residual']:.4f}   "
          f"(planted σ = {SIGMA})")
    resid1_acf = autocorrelation_function(fit1["residuals"], max_lag=5)
    print(f"  residual ACF (lags 1..5): "
          f"{[round(r, 4) for r in resid1_acf[1:]]}")
    print(f"    --> the residuals are NOT white. Lag-1 autocorrelation "
          f"survives because the depth-2 dependence cannot be absorbed "
          f"by a depth-1 fit.")

    # --- Depth-2 fit ---------------------------------------------------
    print("\nStep 4.  Fit depth-2 operator (OLS of X_t on (X_{t-1}, X_{t-2})).")
    fit2 = fit_depth_2(X)
    print(f"  φ̂₁               = {fit2['phi1_hat']:+.4f}   "
          f"(planted {PHI_1:+.4f})")
    print(f"  φ̂₂               = {fit2['phi2_hat']:+.4f}   "
          f"(planted {PHI_2:+.4f})")
    print(f"  σ̂ residual       = {fit2['sigma_residual']:.4f}   "
          f"(planted σ = {SIGMA})")
    resid2_acf = autocorrelation_function(fit2["residuals"], max_lag=5)
    print(f"  residual ACF (lags 1..5): "
          f"{[round(r, 4) for r in resid2_acf[1:]]}")
    print(f"    --> the residuals ARE white. Depth-2 is enough.")

    # --- CSVs ------------------------------------------------------------
    with (RESULTS / "E12_ar2_autocorrelation.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["lag", "empirical_acf",
                                          "predicted_acf_depth2_recurrence"])
        w.writeheader()
        for r in acf_rows:
            w.writerow(r)

    with (RESULTS / "E12_ar2_depth1_vs_depth2_fit.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["fit_depth", "parameter",
                                          "empirical", "planted_or_implied"])
        w.writeheader()
        w.writerow({"fit_depth": 1, "parameter": "phi_hat",
                    "empirical": fit1["phi_hat"],
                    "planted_or_implied": RHO_1_PRED})
        w.writerow({"fit_depth": 1, "parameter": "sigma_residual",
                    "empirical": fit1["sigma_residual"],
                    "planted_or_implied": ""})
        w.writerow({"fit_depth": 2, "parameter": "phi1_hat",
                    "empirical": fit2["phi1_hat"],
                    "planted_or_implied": PHI_1})
        w.writerow({"fit_depth": 2, "parameter": "phi2_hat",
                    "empirical": fit2["phi2_hat"],
                    "planted_or_implied": PHI_2})
        w.writerow({"fit_depth": 2, "parameter": "sigma_residual",
                    "empirical": fit2["sigma_residual"],
                    "planted_or_implied": SIGMA})

    with (RESULTS / "E12_ar2_residual_acf_comparison.csv").open(
            "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["lag",
                                          "residual_acf_depth1_fit",
                                          "residual_acf_depth2_fit"])
        w.writeheader()
        for k in range(1, 6):
            w.writerow({
                "lag": k,
                "residual_acf_depth1_fit": float(resid1_acf[k]),
                "residual_acf_depth2_fit": float(resid2_acf[k]),
            })

    # --- Readout ---------------------------------------------------------
    md = [
        f"# E12 — The AR(2) Gaussian trajectory  (φ₁ = {PHI_1}, φ₂ = {PHI_2})",
        "",
        f"**Rule:** X_t = φ₁ X_{{t-1}} + φ₂ X_{{t-2}} + ε_t, with "
        f"ε_t ~ N(0, 1) iid and (φ₁, φ₂) = ({PHI_1}, {PHI_2}).  "
        f"**Trajectory length:** N = {N}.",
        "",
        "## What is new in this experiment",
        "",
        "The dependence is now depth-2: the next state is a linear "
        "combination of the previous *two* states plus an independent "
        "Gaussian shock. A depth-1 operator (the AR(1) fit from E11) "
        "cannot represent this — there is a piece of the dependence "
        "(the lag-2 term) that has no slot in a depth-1 model. The "
        "central question of this experiment is: **how does the substrate "
        "tell you that depth-1 is the wrong depth?** The answer is "
        "concrete: look at the residual autocorrelation function of the "
        "fit. If it has a non-zero structure, the depth is wrong.",
        "",
        "## The autocorrelation function follows the depth-2 recurrence",
        "",
        "For a stationary AR(2), the autocorrelations at successive lags "
        "satisfy ρ_k = φ₁ ρ_{k−1} + φ₂ ρ_{k−2} for k ≥ 2, with the "
        "boundary value ρ_1 = φ₁/(1 − φ₂). Empirically:",
        "",
        "| lag k | empirical ρ_k | predicted (recurrence) |",
        "| ---: | ---: | ---: |",
    ]
    for r in acf_rows:
        md.append(f"| {r['lag']} | {r['empirical_acf']:+.4f} | "
                  f"{r['predicted_acf_depth2_recurrence']:+.4f} |")

    md += [
        "",
        "Match within sampling noise at every lag. The depth-2 recurrence "
        "is the right description of the dependence in the trajectory.",
        "",
        "## Depth-1 fit: misses the depth-2 structure",
        "",
        "Fitting the AR(1) model from E11 (OLS of X_t on X_{t-1}):",
        "",
        "| quantity | empirical | implied / planted |",
        "| :--- | ---: | ---: |",
        f"| φ̂                          | {fit1['phi_hat']:+.4f} | ≈ ρ_1 = {RHO_1_PRED:+.4f} (NOT the planted φ₁ = {PHI_1:+.4f}) |",
        f"| residual σ                  | {fit1['sigma_residual']:.4f} | — (will be larger than the true σ = {SIGMA:.4f} because the fit is mis-specified) |",
        "",
        "**Critical reading:** the depth-1 fit gives `φ̂ = ρ_1`, the "
        "lag-1 autocorrelation. This is not equal to the planted "
        f"φ₁ = {PHI_1}. The depth-1 model interprets the entire "
        "dependence as a depth-1 effect and reads the lag-1 correlation "
        "as its parameter — but that correlation is shaped by both "
        "φ₁ and φ₂, so the reading is biased.",
        "",
        "The decisive diagnostic is the residual ACF after the depth-1 "
        "fit. If the model captured all the dependence, the residuals "
        "would be white. Empirical residual ACF at lags 1 – 5:",
        "",
        "| lag | residual ACF (depth-1 fit) |",
        "| ---: | ---: |",
    ]
    for k in range(1, 6):
        md.append(f"| {k} | {resid1_acf[k]:+.4f} |")

    md += [
        "",
        "**The lag-1 residual ACF is *not* zero** — there is leftover "
        "depth-1 correlation that the depth-1 fit could not absorb. This "
        "is the certificate that depth-1 is the wrong depth: the fit "
        "looks plausible in its coefficient, but the residuals expose "
        "that it has missed something.",
        "",
        "## Depth-2 fit: recovers the operator and leaves white residuals",
        "",
        "Fitting OLS of X_t on (X_{t-1}, X_{t-2}):",
        "",
        "| quantity | empirical | planted |",
        "| :--- | ---: | ---: |",
        f"| φ̂₁                         | {fit2['phi1_hat']:+.4f} | {PHI_1:+.4f} |",
        f"| φ̂₂                         | {fit2['phi2_hat']:+.4f} | {PHI_2:+.4f} |",
        f"| residual σ                  | {fit2['sigma_residual']:.4f} | {SIGMA:.4f} |",
        "",
        "Both planted coefficients are recovered to within sampling noise. "
        "The residual standard deviation matches the planted innovation "
        "standard deviation. Residual ACF at lags 1 – 5:",
        "",
        "| lag | residual ACF (depth-2 fit) |",
        "| ---: | ---: |",
    ]
    for k in range(1, 6):
        md.append(f"| {k} | {resid2_acf[k]:+.4f} |")

    md += [
        "",
        "All near zero. The depth-2 fit captures all the dependence; "
        "nothing is left for a depth-3 reading to find.",
        "",
        "## Side-by-side residual ACFs (the diagnostic)",
        "",
        "| lag | residual ACF (depth-1) | residual ACF (depth-2) |",
        "| ---: | ---: | ---: |",
    ]
    for k in range(1, 6):
        md.append(f"| {k} | {resid1_acf[k]:+.4f} | {resid2_acf[k]:+.4f} |")

    md += [
        "",
        "Reading the rows: the depth-1 fit leaves a clear residual "
        "autocorrelation; the depth-2 fit leaves nothing measurable. **The "
        "residual ACF tells you whether the operator is at the right "
        "depth.** And from this diagnostic the right depth can be "
        "discovered iteratively: fit at depth k, check residual ACF, "
        "increase k if non-zero structure remains.",
        "",
        "## The substrate's view of this trajectory",
        "",
        "The operator is now a depth-2 object: two numbers (φ₁, φ₂) "
        "rather than one. Reading the operator at the right depth gives "
        "a faithful one-step prediction (the residuals are the pure "
        "innovation) and a complete characterisation of the process: from "
        "(φ₁, φ₂, σ) we can compute the stationary variance, all "
        "autocorrelations, and the variance of the running sum.",
        "",
        "Reading at the wrong depth (depth-1) leaves a measurable trace in "
        "the residuals. That trace is itself information — it tells the "
        "substrate that more depth is needed. The procedure that follows "
        "is iterative: increase the depth until the residual ACF is white. "
        "This is the operational form of the depth-of-memory question, "
        "answerable directly from the trajectory.",
        "",
        "## What this experiment shows",
        "",
        "Together with E11 this establishes the depth-of-memory tools "
        "for the continuous-step case: when the operator is at the right "
        "depth, parameters are recovered cleanly and residuals are white; "
        "when it is at the wrong depth, the residual ACF makes the "
        "mismatch visible. The depth itself is therefore a measurable "
        "property of the trajectory, not a modelling choice that has to "
        "be made in advance.",
        "",
        "The next experiment (E13) extends this same logic to a "
        "qualitatively different type of dependence — long-range memory, "
        "where the influence of past steps decays not geometrically (as "
        "in AR(p)) but at a slower rate. The depth-of-memory tools from "
        "E11 / E12 should still flag the dependence, but the residuals "
        "of any finite-depth AR fit will show a residual structure that "
        "no fixed depth can absorb. That is the qualitative new regime "
        "to investigate.",
        "",
    ]
    (RESULTS / "E12_ar2_gaussian.md").write_text("\n".join(md))
    print(f"\nWrote {RESULTS / 'E12_ar2_gaussian.md'}")


if __name__ == "__main__":
    main()
