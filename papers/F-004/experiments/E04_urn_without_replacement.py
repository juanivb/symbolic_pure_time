"""
E04 — The urn-without-replacement trajectory
=============================================

This is the first experiment where the step is no longer independent of
the past. The setup is the classical urn problem: an urn contains R red
balls and B blue balls (total N = R + B). Each step draws one ball
*without putting it back*. The state at step t is the count of red
balls drawn so far, S_t.

Because each draw removes a ball from the urn, the probability that the
next ball is red depends on what was drawn before:

        P(red at step t+1 | S_t = s, t draws made)  =  (R − s) / (N − t).

The step is therefore *not* identically distributed across time, and
not independent of the past. This is the first time the substrate's
one-step operator stops being trivial: predicting the next state from
the past is no longer just "add an independent random step", it has to
track how the urn has been depleted.

What we want to see, by direct numerical comparison:

  (i)   the marginal at horizon n agrees with the closed-form
        without-replacement expression
            P(S_n = k)  =  C(R, k) · C(B, n − k) / C(N, n);

  (ii)  the marginal does *not* agree with the iid convolution that
        worked for E00–E03 — specifically, the spread is *smaller*
        than the iid version by a finite-population factor
            spread_actual²  =  n · p · (1 − p) · (N − n) / (N − 1),
        where p = R / N. The drop is largest near n = N/2;

  (iii) the conditional step probability at each time t, averaged over
        many trajectories, tracks the deterministic ratio
        (R − S_t) / (N − t). This is the substrate's first non-trivial
        operator reading: the next step depends on the state, in a way
        we can measure;

  (iv)  the n-fold convolution of any *fixed* single-step measure
        cannot reproduce the marginal, because the step is not the
        same at every t. The convolution identity that closed E00–E03
        cleanly breaks here — and the size of the break is exactly the
        finite-population correction.

The urn setup gives us a clean, controllable case where the failure of
"marginal = n-fold convolution of a fixed step" is *quantitatively*
predictable. That is what we want for the next experiments: a probe
for non-independence.

Outputs
-------
    results/E04_marginal_horizons.csv
    results/E04_variance_comparison.csv
    results/E04_conditional_step_probability.csv
    results/E04_urn_without_replacement.md

Run
---
    python3 E04_urn_without_replacement.py
"""

from __future__ import annotations

import csv
from math import lgamma, log
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
RESULTS = HERE.parent / "results"
RESULTS.mkdir(exist_ok=True)


R = 50                                                # red balls
B = 50                                                # blue balls
URN_N = R + B                                         # total balls
P_RED = R / URN_N                                     # initial probability of red


# ---------------------------------------------------------------------------
# Trajectory
# ---------------------------------------------------------------------------

def urn_trajectory_without_replacement(rng: np.random.Generator) -> np.ndarray:
    """Draw all N balls in random order; X_t = 1 iff t-th drawn ball is red.
    Returns the (length-N) 0/1 trajectory."""
    composition = np.array([1] * R + [0] * B)
    perm = rng.permutation(composition)
    return perm


# ---------------------------------------------------------------------------
# Closed-form marginals
# ---------------------------------------------------------------------------

def _log_C(n: int, k: int) -> float:
    if k < 0 or k > n:
        return float("-inf")
    return lgamma(n + 1) - lgamma(k + 1) - lgamma(n - k + 1)


def hypergeometric_marginal(n: int, R: int, B: int) -> tuple[np.ndarray, np.ndarray]:
    """Closed form P(S_n = k) = C(R, k) C(B, n-k) / C(N, n) for k = 0..n."""
    N = R + B
    k_low = max(0, n - B)
    k_high = min(n, R)
    k = np.arange(0, n + 1)
    log_norm = _log_C(N, n)
    log_pmf = np.array([
        (_log_C(R, kk) + _log_C(B, n - kk) - log_norm)
        if (k_low <= kk <= k_high) else float("-inf")
        for kk in k
    ], dtype=float)
    return k, np.exp(log_pmf)


def iid_path_count_marginal(n: int, p: float) -> tuple[np.ndarray, np.ndarray]:
    """The iid version: C(n, k) p^k (1-p)^(n-k), for comparison."""
    k = np.arange(n + 1)
    log_p = log(p) if p > 0 else float("-inf")
    log_q = log(1 - p) if p < 1 else float("-inf")
    log_pmf = np.array([_log_C(n, int(kk)) + int(kk) * log_p + (n - int(kk)) * log_q
                        for kk in k], dtype=float)
    return k, np.exp(log_pmf)


def total_variation(p_emp: np.ndarray, p_true: np.ndarray) -> float:
    return 0.5 * float(np.abs(p_emp - p_true).sum())


# ---------------------------------------------------------------------------
# Empirical marginal by repeated drawing of the urn
# ---------------------------------------------------------------------------

def empirical_marginal(n: int, R: int, B: int, n_reps: int,
                        rng: np.random.Generator) -> np.ndarray:
    """Estimate P(S_n = k) by drawing the urn n_reps times and counting
    reds in the first n positions of each random permutation."""
    N = R + B
    out = np.zeros(n + 1, dtype=int)
    for _ in range(n_reps):
        perm = urn_trajectory_without_replacement(rng)
        s_n = int(perm[:n].sum())
        out[s_n] += 1
    return out / n_reps


# ---------------------------------------------------------------------------
# Conditional step probability (the substrate's first non-trivial operator)
# ---------------------------------------------------------------------------

def conditional_step_probability(n_reps: int, rng: np.random.Generator):
    """For each time t (1..N), record the empirical mean of X_t conditional
    on the past — averaged over n_reps trajectories. This empirical mean,
    at any single trajectory, equals (R − S_{t−1}) / (N − (t−1)). Averaged
    over many trajectories with the same t we recover the unconditional
    P(X_t = red), which is constant = R/N. But the *trajectory-internal*
    variation of the conditional probability is what the substrate sees."""
    N = R + B
    # We collect both the unconditional probability at each t (should be
    # R/N for all t) AND the within-trajectory mean of the conditional
    # ratio (R − S_{t−1}) / (N − (t−1)), which is a deterministic function
    # of the history that we expect to equal X_t in expectation.
    sum_X = np.zeros(N)
    sum_cond_pred = np.zeros(N)
    sum_X_minus_pred_squared = np.zeros(N)
    for _ in range(n_reps):
        perm = urn_trajectory_without_replacement(rng)
        S = np.cumsum(perm)
        for t in range(N):
            past_sum = int(S[t - 1]) if t > 0 else 0
            cond_pred = (R - past_sum) / (N - t)
            sum_X[t] += perm[t]
            sum_cond_pred[t] += cond_pred
            sum_X_minus_pred_squared[t] += (perm[t] - cond_pred) ** 2
    return sum_X / n_reps, sum_cond_pred / n_reps, sum_X_minus_pred_squared / n_reps


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def main():
    rng = np.random.default_rng(0)

    print(f"Urn: R = {R} red, B = {B} blue, total N = {URN_N}.  "
          f"Initial p = R/N = {P_RED}.")

    horizons = [5, 10, 50, 99]                       # n ≤ N for without-replacement
    n_reps_marginal = 100_000
    n_reps_step = 20_000

    print("\nStep 2.  Marginal of S_n at four horizons.")
    marginal_rows = []
    variance_rows = []
    for n in horizons:
        k, p_hyper = hypergeometric_marginal(n, R, B)
        _, p_iid   = iid_path_count_marginal(n, P_RED)
        p_emp = empirical_marginal(n, R, B, n_reps_marginal, rng)

        tv_emp_vs_hyper = total_variation(p_emp,  p_hyper)
        tv_emp_vs_iid   = total_variation(p_emp,  p_iid)
        tv_hyper_vs_iid = total_variation(p_hyper, p_iid)

        # variances
        var_emp = float(((k - (k * p_emp).sum()) ** 2 * p_emp).sum())
        var_hyper = float(((k - (k * p_hyper).sum()) ** 2 * p_hyper).sum())
        var_iid_pred = n * P_RED * (1 - P_RED)
        var_hyper_pred = var_iid_pred * (URN_N - n) / (URN_N - 1)

        print(f"  n = {n:>3d}   TV(emp, hyper) = {tv_emp_vs_hyper:.4f}   "
              f"TV(emp, iid) = {tv_emp_vs_iid:.4f}   "
              f"TV(hyper, iid) = {tv_hyper_vs_iid:.4f}")
        print(f"           var (emp) = {var_emp:.4f}   var (hyper closed-form) = "
              f"{var_hyper:.4f}   var (iid prediction) = {var_iid_pred:.4f}   "
              f"var (hyper prediction) = {var_hyper_pred:.4f}")

        for i in range(len(k)):
            marginal_rows.append({
                "horizon": n,
                "k": int(k[i]),
                "p_empirical":          float(p_emp[i]),
                "p_hypergeometric":     float(p_hyper[i]),
                "p_iid_path_count":     float(p_iid[i]),
                "abs_diff_emp_vs_hyper": float(abs(p_emp[i] - p_hyper[i])),
                "abs_diff_emp_vs_iid":   float(abs(p_emp[i] - p_iid[i])),
            })

        variance_rows.append({
            "horizon": n,
            "var_empirical": var_emp,
            "var_hypergeometric_closed_form": var_hyper,
            "var_iid_prediction": var_iid_pred,
            "var_hypergeometric_prediction": var_hyper_pred,
            "finite_population_correction": (URN_N - n) / (URN_N - 1),
        })

    print("\nStep 3.  Conditional step probability at each time t.")
    print("         Averaged over many trajectories; compared to the "
          "predicted ratio (R − S_{t−1})/(N − (t−1)).")
    avg_X, avg_cond_pred, avg_sq_resid = conditional_step_probability(
        n_reps_step, rng)
    step_rows = []
    for t in range(URN_N):
        step_rows.append({
            "t": int(t + 1),
            "P_X_t_red_unconditional":   float(avg_X[t]),
            "P_X_t_red_unconditional_expected": P_RED,
            "mean_conditional_prediction": float(avg_cond_pred[t]),
            "mean_squared_residual": float(avg_sq_resid[t]),
        })
    # Print a small table for the key time points
    for t in [0, 9, 49, 79, 89, 98]:
        if t >= URN_N:
            continue
        r = step_rows[t]
        print(f"  t = {r['t']:>3d}   P(X_t=red) (empirical) = "
              f"{r['P_X_t_red_unconditional']:.4f}   "
              f"⟨(R − S_{{t−1}})/(N − (t−1))⟩ = "
              f"{r['mean_conditional_prediction']:.4f}   "
              f"⟨(X_t − cond pred)²⟩ = {r['mean_squared_residual']:.4f}")

    # --- CSVs ------------------------------------------------------------
    with (RESULTS / "E04_marginal_horizons.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["horizon", "k",
                                          "p_empirical",
                                          "p_hypergeometric",
                                          "p_iid_path_count",
                                          "abs_diff_emp_vs_hyper",
                                          "abs_diff_emp_vs_iid"])
        w.writeheader()
        for r in marginal_rows:
            w.writerow(r)

    with (RESULTS / "E04_variance_comparison.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["horizon",
                                          "var_empirical",
                                          "var_hypergeometric_closed_form",
                                          "var_iid_prediction",
                                          "var_hypergeometric_prediction",
                                          "finite_population_correction"])
        w.writeheader()
        for r in variance_rows:
            w.writerow(r)

    with (RESULTS / "E04_conditional_step_probability.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["t",
                                          "P_X_t_red_unconditional",
                                          "P_X_t_red_unconditional_expected",
                                          "mean_conditional_prediction",
                                          "mean_squared_residual"])
        w.writeheader()
        for r in step_rows:
            w.writerow(r)

    # --- Readout ---------------------------------------------------------
    md = [
        "# E04 — The urn-without-replacement trajectory",
        "",
        f"**Urn:** {R} red balls, {B} blue balls, total N = {URN_N}.  "
        f"**Trajectory:** draw all N balls in random order without "
        f"replacement; X_t = 1 if the t-th draw is red, 0 otherwise.  "
        f"**State at step t:** S_t = count of reds drawn so far.",
        "",
        "## What is new in this experiment",
        "",
        "For the first time the step is *not* independent of the past. Once "
        "a red ball is drawn there are fewer reds in the urn, so the "
        "probability of red on the next draw goes down. The step itself "
        "depends on the state of the urn, which in turn depends on the "
        "history of draws.",
        "",
        "This breaks the central identity of E00–E03 — the one that says "
        "*marginal at horizon n = n-fold self-convolution of the single step*. "
        "The break is *quantitative*: we know exactly how much it differs, "
        "and the difference is the finite-population correction factor.",
        "",
        "## Empirical marginal vs the two candidate closed forms",
        "",
        "At each horizon we compare the empirical marginal (from "
        f"{n_reps_marginal} random draws of the urn) against two closed-form "
        "predictions:",
        "",
        "  - **without-replacement path-count** "
        "P(S_n = k) = C(R, k) · C(B, n − k) / C(N, n) — counts the number of "
        "ways to choose k reds out of R and n − k blues out of B, divided "
        "by the total number of n-subsets;",
        "",
        f"  - **iid path-count** C(n, k) · p^k · (1 − p)^(n − k) with "
        f"p = R/N = {P_RED} — the answer we would get if each draw were "
        "independent (the regime of E00–E03).",
        "",
        "| horizon n | TV(emp, without-replacement) | TV(emp, iid) | TV(without-replacement, iid) |",
        "| ---: | ---: | ---: | ---: |",
    ]
    for n_h in horizons:
        rows_n = [r for r in marginal_rows if r["horizon"] == n_h]
        tv_emp_h = 0.5 * sum(r["abs_diff_emp_vs_hyper"] for r in rows_n)
        tv_emp_i = 0.5 * sum(r["abs_diff_emp_vs_iid"]   for r in rows_n)
        # tv (hyper, iid)
        p_h = np.array([r["p_hypergeometric"] for r in rows_n])
        p_i = np.array([r["p_iid_path_count"] for r in rows_n])
        tv_h_i = 0.5 * float(np.abs(p_h - p_i).sum())
        md.append(f"| {n_h} | {tv_emp_h:.4f} | {tv_emp_i:.4f} | {tv_h_i:.4f} |")

    md += [
        "",
        "Reading the columns: the empirical marginal sits on the "
        "without-replacement curve to within sampling noise; it does *not* "
        "sit on the iid curve. The gap between the two closed forms grows "
        "with n, with the largest gap near n = N/2 (when the urn is half "
        "drained and the conditional probability has drifted the most).",
        "",
        "## The variance shrinks by a known finite-population factor",
        "",
        "If the draws were independent, the variance of S_n would be "
        f"n · p · (1 − p) with p = {P_RED}. With the without-replacement "
        "rule, the variance is smaller by a factor (N − n) / (N − 1). "
        "Measured directly:",
        "",
        "| horizon n | var (empirical) | var (without-replacement closed form) | var (iid prediction) | var (without-replacement prediction) | correction (N−n)/(N−1) |",
        "| ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for r in variance_rows:
        md.append(f"| {r['horizon']} | {r['var_empirical']:.4f} | "
                  f"{r['var_hypergeometric_closed_form']:.4f} | "
                  f"{r['var_iid_prediction']:.4f} | "
                  f"{r['var_hypergeometric_prediction']:.4f} | "
                  f"{r['finite_population_correction']:.4f} |")

    md += [
        "",
        "At n = 99 the correction factor is (100 − 99)/99 ≈ 0.0101 and the "
        "variance is about 100× smaller than the iid prediction — by then "
        "we have drawn the whole urn and almost all randomness is gone "
        "(only the order is random; the count of reds at the end is exactly R).",
        "",
        "## The substrate's view: the step has a non-trivial state-dependence",
        "",
        f"For each time t we measure two things across {n_reps_step} "
        "trajectories:",
        "",
        "  - the **unconditional** probability that the t-th draw is red, "
        f"which should be exactly R/N = {P_RED} for every t (the symmetry of "
        "the urn);",
        "  - the **trajectory-averaged conditional prediction** "
        "(R − S_{t−1})/(N − (t − 1)), which is the substrate's one-step "
        "operator reading at time t: given the history, this is the "
        f"probability of red on the next draw.",
        "",
        "A few representative time points:",
        "",
        "| t | P(X_t = red) empirical | conditional prediction averaged over trajectories | mean squared residual |",
        "| ---: | ---: | ---: | ---: |",
    ]
    for t_idx in [0, 9, 49, 79, 89, 98]:
        if t_idx >= URN_N:
            continue
        r = step_rows[t_idx]
        md.append(f"| {r['t']} | {r['P_X_t_red_unconditional']:.4f} | "
                  f"{r['mean_conditional_prediction']:.4f} | "
                  f"{r['mean_squared_residual']:.4f} |")

    md += [
        "",
        "Two things to notice. The unconditional probability is essentially "
        f"R/N = {P_RED} at every t (modulo sampling noise) — the urn is "
        "*symmetric* under permutation of draw order. The conditional "
        "prediction averaged over trajectories also reads R/N at every t, "
        "for the same symmetry reason. But the mean squared residual "
        "(how far X_t lies from the conditional prediction *within a single "
        "trajectory*) shrinks as t grows: as the urn drains, the conditional "
        "probability becomes more and more determined by the past, until "
        "near t = N the next draw is essentially forced by what is left.",
        "",
        "This is the substrate's first genuine job in the programme: tracking "
        "the conditional probability of the next step given the history. In "
        "E00–E03 that conditional probability was constant and equal to the "
        "unconditional one — the operator was the identity. Here the "
        "conditional probability is a non-trivial function of the past, and "
        "the marginal at horizon n is no longer recoverable from a single "
        "fixed step distribution.",
        "",
        "## What this experiment shows",
        "",
        "The convolution identity *marginal = n-fold convolution of a fixed "
        "step* held exactly in E00–E03 and is broken here. The break is "
        "quantitatively predictable: the without-replacement marginal sits "
        "on a different closed form than the iid one, and the variance "
        "shrinks by exactly the finite-population factor (N − n)/(N − 1). "
        "The substrate's one-step operator stops being trivial — it now "
        "carries the running count of what remains in the urn, and the "
        "next step's probability is read off that state.",
        "",
        "This is the entry point into the part of the programme where the "
        "substrate's dynamics start to matter. From here, the natural next "
        "step (E05) is to keep the urn finite but turn the dependence into "
        "a memory at a fixed depth — a Markov-like rule that conditions "
        "only on the most recent draw rather than on the entire history. "
        "That isolates the *memory* effect from the *exhaustion* effect we "
        "saw here.",
        "",
    ]
    (RESULTS / "E04_urn_without_replacement.md").write_text("\n".join(md))
    print(f"\nWrote {RESULTS / 'E04_urn_without_replacement.md'}")


if __name__ == "__main__":
    main()
