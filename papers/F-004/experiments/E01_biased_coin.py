"""
E01 — The biased coin trajectory
=================================

The same kind of trajectory as E00, but the coin is no longer fair: it
lands heads with probability p = 0.3 and tails with probability
1 − p = 0.7. The state at step t is the running count of heads
S_t = X_1 + ... + X_t.

The point of this experiment is to see what changes when the single
step loses its left-right symmetry. The expected horizon-n marginal
is no longer centred at n/2 but at np, its spread is √(np(1-p)), and
at finite horizons it carries an asymmetry — the right and left tails
of the discrete distribution are not mirror images of each other. As
the horizon grows the asymmetry of the standardised marginal shrinks
roughly as 1/√n: the smooth bell still wins at large enough n, but
slower than in the symmetric case.

Everything else is structurally identical to E00:

  (i)   the marginal at horizons n = 5, 10, 50, 500 matches the
        closed-form path-count expression P(S_n = k) = C(n, k) p^k (1-p)^(n-k)
        within sampling noise;

  (ii)  the standardised marginal becomes nearly symmetric at large n —
        we report the skewness of the standardised marginal at each
        horizon to see this happen;

  (iii) the substrate's reading: trivial dynamics (independent steps),
        non-trivial innovation (asymmetric step with non-zero third
        moment);

  (iv)  the n-fold convolution of the empirical step equals the
        marginal at each horizon, to floating-point precision when the
        step is the exact (p, 1−p) measure.

Outputs
-------
    results/E01_marginal_horizons.csv
    results/E01_skew_decay.csv
    results/E01_convolution_vs_path_count.csv
    results/E01_innovation_reading.csv
    results/E01_biased_coin.md

Run
---
    python3 E01_biased_coin.py
"""

from __future__ import annotations

import csv
from math import comb
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
RESULTS = HERE.parent / "results"
RESULTS.mkdir(exist_ok=True)


P_HEADS = 0.3                                          # probability of heads
P_TAILS = 1.0 - P_HEADS


# ---------------------------------------------------------------------------
# Step 1.  Generate the trajectory.
# ---------------------------------------------------------------------------

def biased_coin_trajectory(N: int, p: float, rng: np.random.Generator):
    X = (rng.random(size=N) < p).astype(np.int64)
    S = np.cumsum(X)
    return X, S


# ---------------------------------------------------------------------------
# Step 2.  The marginal at each horizon, two ways.
# ---------------------------------------------------------------------------

def closed_form_marginal(n: int, p: float) -> tuple[np.ndarray, np.ndarray]:
    k = np.arange(n + 1)
    coef = np.array([comb(n, kk) for kk in k], dtype=float)
    prob = coef * (p ** k) * ((1.0 - p) ** (n - k))
    return k, prob


def empirical_marginal(n: int, p: float, n_reps: int,
                       rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    outcomes = (rng.random((n_reps, n)) < p).astype(np.int64)
    s_n = outcomes.sum(axis=1)
    k = np.arange(n + 1)
    counts = np.bincount(s_n, minlength=n + 1)
    return k, counts / counts.sum()


def total_variation(p_emp: np.ndarray, p_true: np.ndarray) -> float:
    return 0.5 * float(np.abs(p_emp - p_true).sum())


def standardised_skew_of_marginal(n: int, p: float) -> float:
    """Closed-form skewness of (S_n - np)/√(np(1-p))."""
    return float((1.0 - 2.0 * p) / np.sqrt(n * p * (1.0 - p)))


# ---------------------------------------------------------------------------
# Step 3.  The substrate's view: dynamics + innovation.
# ---------------------------------------------------------------------------

def innovation_reading(X: np.ndarray, p: float) -> dict:
    """The centred coin step ε = X − p, read directly from the sample."""
    eps = X.astype(float) - p
    vals, counts = np.unique(eps, return_counts=True)
    probs = counts / counts.sum()
    var_expected = p * (1.0 - p)
    third_expected = p * (1.0 - p) * (1.0 - 2.0 * p)
    return {
        "support": [float(v) for v in vals],
        "empirical_probabilities": [float(pp) for pp in probs],
        "mean": float(eps.mean()),
        "variance": float(eps.var()),
        "third_moment": float((eps ** 3).mean()),
        "fourth_moment": float((eps ** 4).mean()),
        "variance_expected": var_expected,
        "third_moment_expected": third_expected,
    }


def nfold_convolution_binary(values: np.ndarray, probs: np.ndarray, n: int):
    v0, v1 = float(values[0]), float(values[1])
    base = np.array([float(probs[0]), float(probs[1])])
    out = base.copy()
    for _ in range(n - 1):
        out = np.convolve(out, base)
    k = np.arange(n + 1)
    support = k * v1 + (n - k) * v0
    return support, out


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def main():
    rng = np.random.default_rng(0)
    N = 10_000

    print(f"Step 1.  Generate a biased coin trajectory of length {N} with "
          f"P(heads) = {P_HEADS}")
    X, S = biased_coin_trajectory(N, P_HEADS, rng)
    print(f"  fraction of heads = {X.mean():.4f}   (expected {P_HEADS})")
    print(f"  final running sum S_N = {S[-1]}   "
          f"(expected ~ {N*P_HEADS:.0f} ± {np.sqrt(N*P_HEADS*P_TAILS):.1f})")

    # --- Marginal at horizons -------------------------------------------
    print("\nStep 2.  Marginal of S_n at four horizons.")
    horizons = [5, 10, 50, 500]
    n_reps = 200_000
    marginal_rows = []
    skew_rows = []
    for n in horizons:
        _, p_true = closed_form_marginal(n, P_HEADS)
        _, p_emp = empirical_marginal(n, P_HEADS, n_reps, rng)
        tv = total_variation(p_emp, p_true)
        skew_pred = standardised_skew_of_marginal(n, P_HEADS)

        # Empirical skewness of the standardised marginal.
        k = np.arange(n + 1)
        mu = n * P_HEADS
        sigma = np.sqrt(n * P_HEADS * P_TAILS)
        z = (k - mu) / sigma
        skew_emp_from_path = float(np.sum(p_true * z ** 3))

        print(f"  n = {n:>4d}   TV distance = {tv:.4f}   "
              f"closed-form skew of standardised marginal = "
              f"{skew_pred:+.4f}   (cross-check from path-count: "
              f"{skew_emp_from_path:+.4f})")

        skew_rows.append({
            "horizon": n,
            "closed_form_skew_of_standardised_marginal": skew_pred,
            "path_count_skew_cross_check": skew_emp_from_path,
        })

        for kk in range(n + 1):
            marginal_rows.append({
                "horizon": n,
                "k": int(kk),
                "p_path_count": float(p_true[kk]),
                "p_empirical": float(p_emp[kk]),
                "abs_diff": float(abs(p_emp[kk] - p_true[kk])),
            })

    # --- Innovation reading ----------------------------------------------
    print("\nStep 3.  The substrate's view: innovation reading.")
    reading = innovation_reading(X, P_HEADS)
    print(f"  innovation support      = {reading['support']}")
    print(f"  empirical probabilities = {[round(p, 4) for p in reading['empirical_probabilities']]}   "
          f"(expected [{P_TAILS}, {P_HEADS}])")
    print(f"  innovation mean         = {reading['mean']:+.6f}   (expected 0)")
    print(f"  innovation variance     = {reading['variance']:.6f}   "
          f"(expected {reading['variance_expected']:.6f} = p(1-p))")
    print(f"  third moment            = {reading['third_moment']:+.6f}   "
          f"(expected {reading['third_moment_expected']:+.6f} = p(1-p)(1-2p))")
    print(f"  fourth moment           = {reading['fourth_moment']:.6f}")

    # --- Convolution check ----------------------------------------------
    print("\nStep 4.  Marginal at horizon n vs n-fold convolution of the step.")
    support = np.array(reading["support"])
    probs_emp   = np.array(reading["empirical_probabilities"])
    probs_exact = np.array([P_TAILS, P_HEADS])
    conv_rows = []
    for n in horizons:
        _, pmf_emp   = nfold_convolution_binary(support, probs_emp,   n)
        _, pmf_exact = nfold_convolution_binary(support, probs_exact, n)
        _, p_true    = closed_form_marginal(n, P_HEADS)
        l1_exact = float(np.abs(pmf_exact - p_true).sum())
        l1_emp   = float(np.abs(pmf_emp   - p_true).sum())
        print(f"  n = {n:>4d}   L¹ (exact step)     = {l1_exact:.2e}   "
              f"L¹ (empirical step) = {l1_emp:.2e}")
        for i in range(n + 1):
            conv_rows.append({
                "horizon": n,
                "k": int(i),
                "p_convolution_exact":     float(pmf_exact[i]),
                "p_convolution_empirical": float(pmf_emp[i]),
                "p_path_count":            float(p_true[i]),
                "abs_diff_exact":          float(abs(pmf_exact[i] - p_true[i])),
                "abs_diff_empirical":      float(abs(pmf_emp[i]   - p_true[i])),
            })

    # --- CSVs ------------------------------------------------------------
    with (RESULTS / "E01_marginal_horizons.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["horizon", "k", "p_path_count",
                                          "p_empirical", "abs_diff"])
        w.writeheader()
        for r in marginal_rows:
            w.writerow(r)

    with (RESULTS / "E01_skew_decay.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["horizon",
                                          "closed_form_skew_of_standardised_marginal",
                                          "path_count_skew_cross_check"])
        w.writeheader()
        for r in skew_rows:
            w.writerow(r)

    with (RESULTS / "E01_innovation_reading.csv").open("w", newline="") as f:
        keys = ["mean", "variance", "third_moment", "fourth_moment",
                "variance_expected", "third_moment_expected"]
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerow({k: reading[k] for k in keys})

    with (RESULTS / "E01_convolution_vs_path_count.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["horizon", "k",
                                          "p_convolution_exact",
                                          "p_convolution_empirical",
                                          "p_path_count",
                                          "abs_diff_exact",
                                          "abs_diff_empirical"])
        w.writeheader()
        for r in conv_rows:
            w.writerow(r)

    # --- Readout ---------------------------------------------------------
    md = [
        f"# E01 — The biased coin trajectory  (p = {P_HEADS})",
        "",
        f"**Sample length:** N = {N}.  **Probability of heads:** p = {P_HEADS}.  "
        f"**Closed-form marginal at each horizon:** "
        f"P(S_n = k) = C(n, k) · p^k · (1−p)^(n−k)  "
        "(the same path-count expression as for the fair coin, with the paths "
        "now weighted by the probability of their head-tail composition).",
        "",
        "## The marginal is no longer symmetric at finite horizons",
        "",
        "When the coin is fair the marginal at any horizon is left-right "
        "symmetric: P(S_n = k) = P(S_n = n − k). With p = 0.3 that mirror "
        "symmetry breaks. The marginal is still concentrated around its "
        "mean (now n·p, not n/2), its spread is still √(n·p·(1−p)), but "
        "the right tail is shorter than the left tail at every finite n.",
        "",
        "The asymmetry of the standardised marginal "
        "(S_n − n·p) / √(n·p·(1−p)) shrinks as the horizon grows. We report "
        "two cross-checks of this at each horizon: the closed-form skew "
        "(1−2p)/√(n·p·(1−p)) and a direct calculation of the same skew "
        "from the path-count expression itself. They agree.",
        "",
        "| horizon n | skew of standardised marginal (closed form) | cross-check from path-count |",
        "| ---: | ---: | ---: |",
    ]
    for r in skew_rows:
        md.append(f"| {r['horizon']} | "
                  f"{r['closed_form_skew_of_standardised_marginal']:+.4f} | "
                  f"{r['path_count_skew_cross_check']:+.4f} |")

    md += [
        "",
        "The asymmetry decays roughly as 1/√n. At horizon 500 the "
        "standardised marginal is nearly symmetric.",
        "",
        "## Empirical marginals vs path-count",
        "",
        "| horizon n | total-variation distance (empirical vs path-count) | n_reps |",
        "| ---: | ---: | ---: |",
    ]
    for n_h in horizons:
        rows_n = [r for r in marginal_rows if r["horizon"] == n_h]
        tv = 0.5 * sum(r["abs_diff"] for r in rows_n)
        md.append(f"| {n_h} | {tv:.4f} | {n_reps} |")

    md += [
        "",
        "## The substrate's view of this trajectory",
        "",
        "The trajectory has no memory: each flip is independent of the past. "
        "The one-step operator on the jet is the cumulative identity. **All "
        "the substantive content of the trajectory lives in the step itself**, "
        "which is now asymmetric.",
        "",
        "Reading the step (centred to mean zero) from the sample:",
        "",
        "| quantity | value | expected |",
        "| :--- | ---: | ---: |",
        f"| support of the step                  | {reading['support']} | (−p, 1−p) = ({-P_HEADS}, {P_TAILS}) |",
        f"| empirical probabilities              | {[round(p, 4) for p in reading['empirical_probabilities']]} | ({P_TAILS}, {P_HEADS}) |",
        f"| mean                                  | {reading['mean']:+.6f} | 0 |",
        f"| variance                              | {reading['variance']:.6f} | {reading['variance_expected']:.6f} |",
        f"| third moment                          | {reading['third_moment']:+.6f} | {reading['third_moment_expected']:+.6f} |",
        f"| fourth moment                         | {reading['fourth_moment']:.6f} | — |",
        "",
        "**The third moment of the step is non-zero**, which is the first "
        "departure from E00. It is exactly p·(1−p)·(1−2p), and it carries the "
        "asymmetry of the marginal at finite horizons through the convolution.",
        "",
        "## The marginal at horizon n is the n-fold convolution of the step",
        "",
        f"Computing the n-fold self-convolution of the step distribution and "
        f"comparing to the closed-form path-count expression at each horizon, "
        f"with two versions of the step: the exact ({P_TAILS}, {P_HEADS}) "
        f"measure and the empirical measure read from this particular sample "
        f"({reading['empirical_probabilities'][0]:.3f} / "
        f"{reading['empirical_probabilities'][1]:.3f}).",
        "",
        "| horizon n | L¹ (exact-step convolution vs path-count) | L¹ (empirical-step convolution vs path-count) |",
        "| ---: | ---: | ---: |",
    ]
    for n_h in horizons:
        rows_n = [r for r in conv_rows if r["horizon"] == n_h]
        l1_exact = sum(r["abs_diff_exact"]     for r in rows_n)
        l1_emp   = sum(r["abs_diff_empirical"] for r in rows_n)
        md.append(f"| {n_h} | {l1_exact:.2e} | {l1_emp:.2e} |")

    md += [
        "",
        "With the exact step the convolution matches the path-count expression "
        "to floating-point precision: the marginal at any horizon is, "
        "*exactly*, the n-fold convolution of the single-step measure. The "
        "asymmetry of the step is carried faithfully through the convolution "
        "into the asymmetry of the marginal at finite n.",
        "",
        "## What this experiment shows",
        "",
        "The whole story of E00 carries over with one new ingredient. The step "
        "now has a non-zero third moment; the marginal at finite horizons is "
        "visibly asymmetric; the asymmetry of the standardised marginal "
        "decays as 1/√n; the convolution identity still holds exactly. The "
        "smooth bell still appears at large enough n — but the path to it "
        "passes through an asymmetric regime that the symmetric coin did not "
        "have to traverse.",
        "",
    ]
    (RESULTS / "E01_biased_coin.md").write_text("\n".join(md))
    print(f"\nWrote {RESULTS / 'E01_biased_coin.md'}")


if __name__ == "__main__":
    main()
