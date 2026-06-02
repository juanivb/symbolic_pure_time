"""
E08 — The sum-of-exponentials trajectory
==========================================

A third continuous step in the programme. Each step is drawn from an
exponential distribution with rate 1 — strictly positive, with density
f(y) = exp(−y) on [0, ∞). The state at step t is the running sum
S_t = Y_1 + ... + Y_t.

The marginal at horizon n is the Gamma(n, 1) distribution:

        f_Γ(n)(x)  =  x^(n − 1) · exp(−x) / (n − 1)!     for x ≥ 0.

This is the third classical continuous family we recover by
accumulating a transformed Gaussian-related step, after the Gaussian
itself (E06) and the chi-squared (E07).

The step has four moments that we read directly from the sample, and
the marginal at horizon n inherits them through the convolution
identity exactly as in earlier experiments:

  step mean        = 1            →  marginal mean        = n
  step variance    = 1            →  marginal variance    = n
  step skewness    = 2            →  marginal skewness    = 2/√n
  step excess kurt = 6            →  marginal excess kurt = 6/n

The shape-decay rates of skew (1/√n) and excess kurtosis (1/n) match
the rule from note 02 once again, this time with starting constants 2
and 6 — distinct from the χ²(n) case of E07 where they were √8 and
12.

Outputs
-------
    results/E08_marginal_horizons.csv
    results/E08_shape_decay.csv
    results/E08_step_reading.csv
    results/E08_sum_of_exponentials.md

Run
---
    python3 E08_sum_of_exponentials.py
"""

from __future__ import annotations

import csv
from math import lgamma
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
RESULTS = HERE.parent / "results"
RESULTS.mkdir(exist_ok=True)


# ---------------------------------------------------------------------------
# Closed form: the Gamma(n, 1) density.
# ---------------------------------------------------------------------------

def gamma_pdf(x: np.ndarray, n: int) -> np.ndarray:
    """f_Γ(n,1)(x) = x^(n−1) · exp(−x) / Γ(n)  for x ≥ 0."""
    out = np.zeros_like(x, dtype=float)
    pos = x > 0
    if not np.any(pos):
        return out
    log_pdf = (n - 1) * np.log(x[pos]) - x[pos] - lgamma(n)
    out[pos] = np.exp(log_pdf)
    return out


# ---------------------------------------------------------------------------
# Empirical marginal: n iid exponentials summed.
# ---------------------------------------------------------------------------

def empirical_gamma_sum(n: int, n_reps: int,
                        rng: np.random.Generator) -> np.ndarray:
    Y = rng.exponential(1.0, size=(n_reps, n))
    return Y.sum(axis=1)


def histogram_vs_density(s_n: np.ndarray, n: int, bins: int = 80) -> dict:
    counts, edges = np.histogram(s_n, bins=bins, density=True)
    centres = 0.5 * (edges[:-1] + edges[1:])
    pdf = gamma_pdf(centres, n)
    return {"centres": centres, "hist_density": counts,
            "pdf_analytical": pdf, "bin_width": float(edges[1] - edges[0])}


def l1_density_distance(h: dict) -> float:
    return float(np.sum(np.abs(h["hist_density"] - h["pdf_analytical"])
                        * h["bin_width"]))


def standardised_skew_kurt(s_n: np.ndarray) -> tuple[float, float]:
    mu = float(s_n.mean())
    sd = float(s_n.std())
    if sd <= 0:
        return 0.0, 0.0
    skew = float(((s_n - mu) ** 3).mean() / sd ** 3)
    ex_kurt = float(((s_n - mu) ** 4).mean() / sd ** 4) - 3.0
    return skew, ex_kurt


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def main():
    rng = np.random.default_rng(0)

    print("Step 1.  Read the exponential step Y ~ Exp(1) from a fresh sample.")
    N_step = 200_000
    Y = rng.exponential(1.0, size=N_step)
    y_mean = float(Y.mean())
    y_var = float(Y.var())
    y_sd = float(np.sqrt(y_var))
    y_skew = float(((Y - y_mean) ** 3).mean() / y_sd ** 3) if y_sd > 0 else 0.0
    y_ek = (float(((Y - y_mean) ** 4).mean() / y_var ** 2) - 3.0
            if y_var > 0 else 0.0)
    print(f"  step mean        = {y_mean:.4f}  (expected 1)")
    print(f"  step variance    = {y_var:.4f}  (expected 1)")
    print(f"  step skewness    = {y_skew:.4f}  (expected 2)")
    print(f"  step excess kurt = {y_ek:.4f}  (expected 6)")

    horizons = [1, 2, 10, 100]
    n_reps = 200_000

    print("\nStep 2.  Marginal of S_n at each horizon vs analytical Gamma(n, 1).")
    marginal_rows = []
    shape_rows = []
    for n in horizons:
        sample = empirical_gamma_sum(n, n_reps, rng)
        mean_emp = float(sample.mean())
        var_emp = float(sample.var())
        skew_emp, ek_emp = standardised_skew_kurt(sample)
        skew_pred = 2.0 / np.sqrt(n)
        ek_pred = 6.0 / n
        h = histogram_vs_density(sample, n)
        l1 = l1_density_distance(h)
        print(f"  n = {n:>4d}   mean (emp) = {mean_emp:>8.4f} (predicted {n})   "
              f"var (emp) = {var_emp:>8.4f} (predicted {n})")
        print(f"            skew (emp) = {skew_emp:.4f} (predicted "
              f"{skew_pred:.4f})   excess kurt (emp) = {ek_emp:.4f} "
              f"(predicted {ek_pred:.4f})   L¹ = {l1:.4f}")

        for i in range(len(h["centres"])):
            marginal_rows.append({
                "horizon": n,
                "x": float(h["centres"][i]),
                "empirical_density": float(h["hist_density"][i]),
                "analytical_gamma_density": float(h["pdf_analytical"][i]),
                "abs_diff": float(abs(h["hist_density"][i] -
                                       h["pdf_analytical"][i])),
            })
        shape_rows.append({
            "horizon": n,
            "mean_empirical": mean_emp,
            "mean_predicted": float(n),
            "var_empirical": var_emp,
            "var_predicted": float(n),
            "skew_empirical": skew_emp,
            "skew_predicted_2_over_sqrt_n": skew_pred,
            "excess_kurt_empirical": ek_emp,
            "excess_kurt_predicted_6_over_n": ek_pred,
            "l1_density_distance": l1,
        })

    # --- CSVs ------------------------------------------------------------
    with (RESULTS / "E08_marginal_horizons.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["horizon", "x",
                                          "empirical_density",
                                          "analytical_gamma_density",
                                          "abs_diff"])
        w.writeheader()
        for r in marginal_rows:
            w.writerow(r)

    with (RESULTS / "E08_shape_decay.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["horizon",
                                          "mean_empirical", "mean_predicted",
                                          "var_empirical", "var_predicted",
                                          "skew_empirical",
                                          "skew_predicted_2_over_sqrt_n",
                                          "excess_kurt_empirical",
                                          "excess_kurt_predicted_6_over_n",
                                          "l1_density_distance"])
        w.writeheader()
        for r in shape_rows:
            w.writerow(r)

    with (RESULTS / "E08_step_reading.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["quantity", "empirical", "expected"])
        w.writeheader()
        for k, e, x in [("mean", y_mean, 1.0),
                        ("variance", y_var, 1.0),
                        ("skewness", y_skew, 2.0),
                        ("excess_kurtosis", y_ek, 6.0)]:
            w.writerow({"quantity": k, "empirical": e, "expected": x})

    # --- Readout ---------------------------------------------------------
    md = [
        "# E08 — The sum-of-exponentials trajectory",
        "",
        "**Step rule:** Y_t ~ Exp(1) independent across t.  "
        "**Trajectory:** S_t = Y_1 + ... + Y_t.  "
        "**Analytical marginal at horizon n:** the Gamma(n, 1) density "
        "x^(n − 1) · exp(−x) / Γ(n) for x ≥ 0.",
        "",
        "## What is new in this experiment",
        "",
        "A third continuous, asymmetric, positive-support step — the "
        "exponential. The accumulation of n such steps gives the Gamma "
        "family at integer shape parameter, which together with E06 "
        "(Gaussian) and E07 (chi-squared) gives us three of the classical "
        "continuous distributions of probability as direct outcomes of the "
        "same accumulation machinery.",
        "",
        "## The step Y ~ Exp(1) and its four moments",
        "",
        "Reading the four leading moments of the step directly from the "
        "sample:",
        "",
        "| quantity | empirical | expected |",
        "| :--- | ---: | ---: |",
        f"| mean             | {y_mean:.4f} | 1 |",
        f"| variance         | {y_var:.4f} | 1 |",
        f"| skewness         | {y_skew:.4f} | 2 |",
        f"| excess kurtosis  | {y_ek:.4f} | 6 |",
        "",
        "## Empirical marginal vs analytical Gamma(n, 1)",
        "",
        f"At each horizon n we draw {n_reps} fresh trajectories, accumulate "
        "n exponential steps in each, and compare the resulting histogram "
        "against the Gamma(n, 1) density.",
        "",
        "| horizon n | mean (emp) | mean (n) | var (emp) | var (n) | L¹ (hist vs density) |",
        "| ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for r in shape_rows:
        md.append(f"| {r['horizon']} | "
                  f"{r['mean_empirical']:.4f} | "
                  f"{r['mean_predicted']:.0f} | "
                  f"{r['var_empirical']:.4f} | "
                  f"{r['var_predicted']:.0f} | "
                  f"{r['l1_density_distance']:.4f} |")

    md += [
        "",
        "Mean and variance both grow as n (the step has mean 1 and variance "
        "1, and the convolution identity adds them across n steps). The "
        "empirical histogram sits on the analytical density at "
        "sampling-noise level.",
        "",
        "## Shape decay",
        "",
        "The step has skewness 2 and excess kurtosis 6. By the rule from "
        "note 02, the marginal at horizon n should have skewness 2/√n "
        "and excess kurtosis 6/n.",
        "",
        "| horizon n | skew (emp) | skew (2/√n) | excess kurt (emp) | excess kurt (6/n) |",
        "| ---: | ---: | ---: | ---: | ---: |",
    ]
    for r in shape_rows:
        md.append(f"| {r['horizon']} | "
                  f"{r['skew_empirical']:+.4f} | "
                  f"{r['skew_predicted_2_over_sqrt_n']:+.4f} | "
                  f"{r['excess_kurt_empirical']:+.4f} | "
                  f"{r['excess_kurt_predicted_6_over_n']:+.4f} |")

    md += [
        "",
        "At n = 1 the marginal is the step itself, with skewness 2 and "
        "excess kurtosis 6. At n = 100 those have shrunk to ≈ 0.2 and "
        "≈ 0.06 respectively — the Gamma(n, 1) marginal is now nearly "
        "bell-shaped. The same rule applies as in E01, E02, E07: each "
        "feature of the step decays at its own 1/n^(m/2 − 1) rate, and "
        "the leading constants 2 and 6 are the only specific marks the "
        "exponential step leaves.",
        "",
        "## The substrate's view of this trajectory",
        "",
        "Independent step → operator is the cumulative identity. All "
        "content lives in the step density f_Y(y) = exp(−y) on [0, ∞). "
        "The marginal at any horizon is the n-fold self-convolution of "
        "this density, with closed form Gamma(n, 1).",
        "",
        "## What this experiment shows",
        "",
        "Three continuous distributions of probability now appear as "
        "outcomes of the same accumulation, applied to three steps:",
        "",
        "  - E06: Gaussian step                  → Gaussian marginal at any horizon.",
        "  - E07: Gaussian step, squared          → chi-squared marginal.",
        "  - E08: exponential step                → Gamma marginal.",
        "",
        "Each step has its own four-moment fingerprint, and the marginal "
        "at horizon n carries that fingerprint scaled by the n^(1 − m/2) "
        "decay. Nothing is postulated; everything follows from the step.",
        "",
    ]
    (RESULTS / "E08_sum_of_exponentials.md").write_text("\n".join(md))
    print(f"\nWrote {RESULTS / 'E08_sum_of_exponentials.md'}")


if __name__ == "__main__":
    main()
