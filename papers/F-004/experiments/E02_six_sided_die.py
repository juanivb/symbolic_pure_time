"""
E02 — The six-sided die trajectory
====================================

The same kind of trajectory as E00 and E01, but the step is now a roll of
a fair six-sided die: a uniform draw from {1, 2, 3, 4, 5, 6}, each value
with probability 1/6. The state at step t is the running sum
S_t = X_1 + ... + X_t.

The structural differences from the coin trajectories of E00/E01:

  • The step has six values rather than two. The support of the marginal
    at horizon n is no longer the n+1 integers between 0 and n but the
    5n+1 integers between n and 6n. The convolution still produces it
    exactly.

  • The step is symmetric about its mean 3.5, so the skewness of the
    step is zero and the skewness of the marginal stays zero at every
    horizon. The familiar "the asymmetric coin's skew decays as 1/√n"
    becomes here "there is no asymmetry to decay at all".

  • The step has a non-zero *excess kurtosis* (it is flatter than a
    bell of the same width: a uniform-on-six). That excess decays at a
    different rate in the marginal as the horizon grows: roughly 1/n
    rather than 1/√n. The shape of the step still leaves a fingerprint
    on the marginal at finite n; the fingerprint just fades faster than
    the skewness one did in E01.

  • The triangular shape of the two-die sum (the iconic peak at 7) is
    just the 2-fold convolution of the uniform-on-six. The shape comes
    out of the convolution by direct computation — no further input.

Everything else carries over from E01 mechanically.

Outputs
-------
    results/E02_marginal_horizons.csv
    results/E02_shape_decay.csv
    results/E02_convolution_vs_pathcount.csv
    results/E02_innovation_reading.csv
    results/E02_six_sided_die.md

Run
---
    python3 E02_six_sided_die.py
"""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
RESULTS = HERE.parent / "results"
RESULTS.mkdir(exist_ok=True)


FACES   = np.array([1, 2, 3, 4, 5, 6], dtype=int)
PROBS   = np.full(6, 1.0 / 6.0)
MU_STEP    = float((FACES * PROBS).sum())                            # 3.5
VAR_STEP   = float(((FACES - MU_STEP) ** 2 * PROBS).sum())           # 35/12
THIRD_STEP = float(((FACES - MU_STEP) ** 3 * PROBS).sum())           # 0
FOURTH_STEP = float(((FACES - MU_STEP) ** 4 * PROBS).sum())          # 14.7292
EXCESS_KURT_STEP = FOURTH_STEP / VAR_STEP ** 2 - 3.0                 # ≈ -1.2686


# ---------------------------------------------------------------------------
# Step 1.  Generate the trajectory.
# ---------------------------------------------------------------------------

def die_trajectory(N: int, rng: np.random.Generator):
    X = rng.integers(1, 7, size=N).astype(np.int64)
    S = np.cumsum(X)
    return X, S


# ---------------------------------------------------------------------------
# Step 2.  The marginal at each horizon, two ways.
# ---------------------------------------------------------------------------

def empirical_marginal(n: int, n_reps: int, rng: np.random.Generator):
    """Estimate P(S_n = k) by fresh sampling of n_reps trajectories of
    length n; k ranges over the integers from n to 6n."""
    outcomes = rng.integers(1, 7, size=(n_reps, n))
    s_n = outcomes.sum(axis=1)
    k_grid = np.arange(n, 6 * n + 1)
    counts = np.bincount(s_n - n, minlength=len(k_grid))[:len(k_grid)]
    return k_grid, counts / counts.sum()


def nfold_convolution_uniform(probs: np.ndarray, n: int):
    """Self-convolve a uniform-on-{1..K} step distribution n times.

    Returns (k_grid, pmf) where k_grid runs over integers from n to K·n
    (K = len(probs)).
    """
    base = np.asarray(probs, dtype=float)
    K = len(base)
    out = base.copy()
    for _ in range(n - 1):
        out = np.convolve(out, base)
    k_grid = np.arange(n, K * n + 1)
    return k_grid, out


def total_variation(p_emp: np.ndarray, p_true: np.ndarray) -> float:
    return 0.5 * float(np.abs(p_emp - p_true).sum())


# ---------------------------------------------------------------------------
# Step 3.  Shape decay (skew and excess kurtosis of the standardised marginal).
# ---------------------------------------------------------------------------

def standardised_skew_and_kurt(k_grid: np.ndarray, pmf: np.ndarray) -> tuple[float, float]:
    """Skewness and excess kurtosis of the distribution with values
    k_grid and probabilities pmf, in standardised units."""
    mu = float((k_grid * pmf).sum())
    var = float(((k_grid - mu) ** 2 * pmf).sum())
    third = float(((k_grid - mu) ** 3 * pmf).sum())
    fourth = float(((k_grid - mu) ** 4 * pmf).sum())
    if var <= 0:
        return 0.0, 0.0
    return third / var ** 1.5, fourth / var ** 2 - 3.0


# ---------------------------------------------------------------------------
# Step 4.  The substrate's view: innovation reading.
# ---------------------------------------------------------------------------

def innovation_reading(X: np.ndarray) -> dict:
    eps = X.astype(float) - MU_STEP
    vals, counts = np.unique(eps, return_counts=True)
    probs = counts / counts.sum()
    return {
        "support": [float(v) for v in vals],
        "empirical_probabilities": [float(p) for p in probs],
        "mean": float(eps.mean()),
        "variance": float(eps.var()),
        "third_moment": float((eps ** 3).mean()),
        "fourth_moment": float((eps ** 4).mean()),
        "variance_expected": VAR_STEP,
        "third_moment_expected": THIRD_STEP,
        "fourth_moment_expected": FOURTH_STEP,
        "excess_kurt_step": (float((eps ** 4).mean())
                             / max(float(eps.var()), 1e-12) ** 2 - 3.0),
        "excess_kurt_step_expected": EXCESS_KURT_STEP,
    }


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def main():
    rng = np.random.default_rng(0)
    N = 10_000

    print(f"Step 1.  Generate a die-roll trajectory of length {N}")
    X, S = die_trajectory(N, rng)
    print(f"  empirical mean of one roll   = {X.mean():.4f}   (expected 3.5)")
    print(f"  empirical variance of one roll = {X.var():.4f}   "
          f"(expected {VAR_STEP:.4f} = 35/12)")
    print(f"  final running sum S_N        = {S[-1]}   "
          f"(expected ~ {N*MU_STEP:.0f} ± {np.sqrt(N*VAR_STEP):.1f})")

    # --- Marginal at horizons -------------------------------------------
    print("\nStep 2.  Marginal of S_n at four horizons.")
    horizons = [1, 2, 5, 50, 500]
    n_reps = 200_000
    marginal_rows = []
    shape_rows = []
    for n in horizons:
        # Path-count via convolution at the exact uniform step
        k_grid, p_true = nfold_convolution_uniform(PROBS, n)
        # Empirical sampling
        k_emp, p_emp = empirical_marginal(n, n_reps, rng)
        tv = total_variation(p_emp, p_true)
        skew, ex_kurt = standardised_skew_and_kurt(k_grid, p_true)
        skew_pred = 0.0                                              # step is symmetric
        ex_kurt_pred = EXCESS_KURT_STEP / n
        print(f"  n = {n:>4d}   TV(emp, path-count) = {tv:.4f}   "
              f"skew = {skew:+.4f} (pred {skew_pred:+.4f})   "
              f"excess kurt = {ex_kurt:+.4f} (pred {ex_kurt_pred:+.4f})")
        shape_rows.append({
            "horizon": n,
            "marginal_skew": skew,
            "marginal_skew_predicted": skew_pred,
            "marginal_excess_kurt": ex_kurt,
            "marginal_excess_kurt_predicted": ex_kurt_pred,
        })
        for kk, p_t, p_e in zip(k_grid, p_true, p_emp):
            marginal_rows.append({
                "horizon": n,
                "k": int(kk),
                "p_path_count": float(p_t),
                "p_empirical": float(p_e),
                "abs_diff": float(abs(p_e - p_t)),
            })

    # --- Innovation reading ---------------------------------------------
    print("\nStep 3.  The substrate's view: innovation reading.")
    reading = innovation_reading(X)
    print(f"  support               = {reading['support']}   "
          f"(expected {[-2.5, -1.5, -0.5, 0.5, 1.5, 2.5]})")
    print(f"  empirical probs       = {[round(p, 4) for p in reading['empirical_probabilities']]}   "
          f"(expected all 1/6 ≈ 0.1667)")
    print(f"  mean                  = {reading['mean']:+.6f}   (expected 0)")
    print(f"  variance              = {reading['variance']:.6f}   "
          f"(expected {VAR_STEP:.6f} = 35/12)")
    print(f"  third moment          = {reading['third_moment']:+.6f}   "
          f"(expected 0, step is symmetric)")
    print(f"  fourth moment         = {reading['fourth_moment']:.6f}   "
          f"(expected {FOURTH_STEP:.6f})")
    print(f"  excess kurtosis (step)= {reading['excess_kurt_step']:+.6f}   "
          f"(expected {EXCESS_KURT_STEP:+.6f})")

    # --- Convolution check ----------------------------------------------
    print("\nStep 4.  Marginal at horizon n vs n-fold convolution of the step.")
    # Empirical probabilities of each face (in the uniform-on-six ordering)
    face_counts = np.array([(X == f).sum() for f in FACES], dtype=float)
    probs_emp = face_counts / face_counts.sum()
    conv_rows = []
    for n in horizons:
        _, pmf_emp   = nfold_convolution_uniform(probs_emp, n)
        _, pmf_exact = nfold_convolution_uniform(PROBS,     n)
        l1_emp = float(np.abs(pmf_emp   - pmf_exact).sum())
        # path-count is by definition the exact convolution; compare to itself
        # gives the floating-point-precision baseline.
        print(f"  n = {n:>4d}   L¹ (empirical step vs exact path-count) = "
              f"{l1_emp:.2e}")
        k_grid, _ = nfold_convolution_uniform(PROBS, n)
        for i, kk in enumerate(k_grid):
            conv_rows.append({
                "horizon": n,
                "k": int(kk),
                "p_convolution_exact":     float(pmf_exact[i]),
                "p_convolution_empirical": float(pmf_emp[i]),
                "abs_diff": float(abs(pmf_emp[i] - pmf_exact[i])),
            })

    # --- CSVs ------------------------------------------------------------
    with (RESULTS / "E02_marginal_horizons.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["horizon", "k", "p_path_count",
                                          "p_empirical", "abs_diff"])
        w.writeheader()
        for r in marginal_rows:
            w.writerow(r)

    with (RESULTS / "E02_shape_decay.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["horizon",
                                          "marginal_skew",
                                          "marginal_skew_predicted",
                                          "marginal_excess_kurt",
                                          "marginal_excess_kurt_predicted"])
        w.writeheader()
        for r in shape_rows:
            w.writerow(r)

    with (RESULTS / "E02_innovation_reading.csv").open("w", newline="") as f:
        keys = ["mean", "variance", "third_moment", "fourth_moment",
                "excess_kurt_step", "variance_expected",
                "third_moment_expected", "fourth_moment_expected",
                "excess_kurt_step_expected"]
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerow({k: reading[k] for k in keys})

    with (RESULTS / "E02_convolution_vs_pathcount.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["horizon", "k",
                                          "p_convolution_exact",
                                          "p_convolution_empirical",
                                          "abs_diff"])
        w.writeheader()
        for r in conv_rows:
            w.writerow(r)

    # --- Readout ---------------------------------------------------------
    md = [
        "# E02 — The six-sided die trajectory",
        "",
        f"**Sample length:** N = {N}.  **Step distribution:** uniform on "
        "{1, 2, 3, 4, 5, 6}.  **Path-count of the marginal at horizon n:** "
        "the n-fold self-convolution of the step. There is no clean compact "
        "formula like C(n, k) / 2ⁿ at this cardinality, but the convolution "
        "is itself the closed form — it counts integer compositions of k "
        "into n parts each in {1..6}, divided by 6ⁿ.",
        "",
        "## What is new in this experiment",
        "",
        "Compared to E00 (fair coin) and E01 (biased coin), the step now has "
        "six values rather than two. The structural changes that follow:",
        "",
        "  - the marginal at horizon n lives on the 5n+1 integers from n to 6n;",
        "  - the step is symmetric about its mean 3.5, so the skewness of the "
        "step is zero — and so is the skewness of the marginal at every "
        "horizon;",
        "  - the step is flatter than a smooth bell of the same width; its "
        "excess kurtosis is "
        f"{EXCESS_KURT_STEP:+.4f}, and this fingerprint decays as roughly "
        "1/n in the marginal as the horizon grows;",
        "  - the n = 2 case is the familiar triangular shape of the sum of "
        "two dice, with peak at 7 — that triangle is what falls out of one "
        "self-convolution of the uniform-on-six.",
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
        "## How the shape of the marginal evolves with horizon",
        "",
        "Skew and excess kurtosis of the standardised marginal at each "
        "horizon, compared to the predicted decay (zero skew at every n, "
        f"excess kurtosis {EXCESS_KURT_STEP:+.4f} / n):",
        "",
        "| horizon n | skew (measured) | skew (predicted) | excess kurt (measured) | excess kurt (predicted) |",
        "| ---: | ---: | ---: | ---: | ---: |",
    ]
    for r in shape_rows:
        md.append(f"| {r['horizon']} | "
                  f"{r['marginal_skew']:+.4f} | "
                  f"{r['marginal_skew_predicted']:+.4f} | "
                  f"{r['marginal_excess_kurt']:+.4f} | "
                  f"{r['marginal_excess_kurt_predicted']:+.4f} |")

    md += [
        "",
        "The skewness is zero at every horizon (by symmetry of the step). "
        "The excess kurtosis at horizon 1 is the step's own kurtosis; it "
        "shrinks as 1/n; by horizon 500 it is below the third decimal place. "
        "The standardised marginal converges to a smooth bell, but the path "
        "passes through a regime where the marginal is symmetric and "
        "flatter than a bell, and then becomes a bell.",
        "",
        "## The substrate's view of this trajectory",
        "",
        "The trajectory has no memory: each roll is independent. The one-step "
        "operator on the jet is the cumulative identity. **All substantive "
        "content lives in the step**, which is now a six-valued symmetric "
        "measure.",
        "",
        "Reading the step (centred to mean zero) from the sample:",
        "",
        "| quantity | value | expected |",
        "| :--- | ---: | ---: |",
        f"| support of the step                  | {reading['support']} | ±2.5, ±1.5, ±0.5 |",
        f"| empirical probabilities              | {[round(p, 4) for p in reading['empirical_probabilities']]} | all 1/6 ≈ 0.1667 |",
        f"| mean                                  | {reading['mean']:+.6f} | 0 |",
        f"| variance                              | {reading['variance']:.6f} | {VAR_STEP:.6f} |",
        f"| third moment                          | {reading['third_moment']:+.6f} | 0 |",
        f"| fourth moment                         | {reading['fourth_moment']:.6f} | {FOURTH_STEP:.6f} |",
        f"| excess kurtosis of step               | {reading['excess_kurt_step']:+.6f} | {EXCESS_KURT_STEP:+.6f} |",
        "",
        "## The marginal at horizon n is the n-fold convolution of the step",
        "",
        "Computing the n-fold self-convolution of the step distribution with "
        "the exact uniform probabilities and with the empirical probabilities "
        "read from this sample, then measuring how far the empirical-step "
        "convolution drifts from the exact one:",
        "",
        "| horizon n | L¹ (empirical-step convolution vs exact-step convolution) |",
        "| ---: | ---: |",
    ]
    for n_h in horizons:
        rows_n = [r for r in conv_rows if r["horizon"] == n_h]
        l1 = sum(r["abs_diff"] for r in rows_n)
        md.append(f"| {n_h} | {l1:.2e} |")

    md += [
        "",
        "The exact-step convolution is itself the closed form at this "
        "cardinality, so the L¹ distance reported here is purely the "
        "amplification of the small sampling noise in the empirical step "
        "probabilities through the n-fold accumulation. It grows with n, as "
        "it should.",
        "",
        "## What this experiment shows",
        "",
        "The path-count = n-fold convolution identity is not a feature of "
        "the binary case. It carries over without modification to the "
        "six-valued step. The shape that emerges at horizon n is now the "
        "convolution of six-valued uniform steps; at n = 2 the convolution "
        "produces the familiar triangle; at moderate n it produces a "
        "bell-with-flat-top; at large n it produces a smooth bell. Each "
        "shape comes out of the same convolution machinery applied to the "
        "step — the substrate has nothing else to add for an independent "
        "trajectory.",
        "",
        "Skewness stays zero (the step has no asymmetry). Excess kurtosis "
        "decays as 1/n. The two decay rates we have now observed are "
        "consistent with a simple rule: the m-th standardised cumulant of "
        "the sum decays as n^{1−m/2}. Skewness (m = 3) goes as 1/√n; "
        "excess kurtosis (m = 4) goes as 1/n. We can verify this rule "
        "directly in future experiments without invoking it as a theorem.",
        "",
    ]
    (RESULTS / "E02_six_sided_die.md").write_text("\n".join(md))
    print(f"\nWrote {RESULTS / 'E02_six_sided_die.md'}")


if __name__ == "__main__":
    main()
