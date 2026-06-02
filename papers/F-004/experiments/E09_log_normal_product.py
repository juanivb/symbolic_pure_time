"""
E09 — The log-normal trajectory  (running product of e^Gaussian steps)
=======================================================================

This is the first experiment in the programme where the accumulation
is not a *sum* but a *product*. Each step is a positive number drawn
from M_t = exp(X_t) with X_t ~ N(0, 1) independent across t. The
state at step t is the running product

        S_t  =  M_1 · M_2 · ... · M_t.

Equivalently, taking the logarithm of both sides:

        log S_t  =  X_1 + X_2 + ... + X_t,

which is exactly the Gaussian random walk of E06. The log-normal
trajectory is the Gaussian walk viewed through an exponential lens.
Whatever holds for the Gaussian walk on the additive side translates
to a corresponding statement on the multiplicative side.

What we want to see:

  (i)   the empirical marginal of S_n at horizons n = 1, 2, 10, 100
        agrees with the closed-form log-normal density of parameters
        (μ = 0, σ² = n) — that is, the density of exp(W) where
        W ~ N(0, n);

  (ii)  the moments of S_n on the *multiplicative* side do *not*
        decay with n — they actually grow exponentially (the log-
        normal has all moments, but they amplify with σ²);

  (iii) the moments of log S_n on the *additive* side recover the
        Gaussian-walk behaviour of E06: variance grows as n, skew and
        excess kurtosis stay near zero.

This is the cleanest demonstration that the shape-decay rule from
note 02 applies to the *additive* accumulation of a step. When the
accumulation is multiplicative, the same rule applies on the
logarithm — and the multiplicative side inherits a corresponding
"growth-of-shape" behaviour that is its mirror image.

Outputs
-------
    results/E09_marginal_horizons.csv
    results/E09_two_sides.csv
    results/E09_log_normal_product.md

Run
---
    python3 E09_log_normal_product.py
"""

from __future__ import annotations

import csv
from math import log
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
RESULTS = HERE.parent / "results"
RESULTS.mkdir(exist_ok=True)


# ---------------------------------------------------------------------------
# Closed form: the log-normal density with parameters (μ = 0, σ² = n).
# ---------------------------------------------------------------------------

def lognormal_pdf(x: np.ndarray, sigma_squared: float) -> np.ndarray:
    """Density of exp(W) where W ~ N(0, sigma_squared)."""
    out = np.zeros_like(x, dtype=float)
    pos = x > 0
    if not np.any(pos):
        return out
    log_x = np.log(x[pos])
    out[pos] = (np.exp(-0.5 * log_x ** 2 / sigma_squared)
                / (x[pos] * np.sqrt(2.0 * np.pi * sigma_squared)))
    return out


def lognormal_moments(sigma_squared: float) -> dict:
    """Closed-form moments of LogNormal(μ = 0, σ² = sigma_squared)."""
    s2 = sigma_squared
    mean = np.exp(s2 / 2.0)
    var = (np.exp(s2) - 1.0) * np.exp(s2)
    if s2 > 0:
        skew = (np.exp(s2) + 2.0) * np.sqrt(np.exp(s2) - 1.0)
        ex_kurt = (np.exp(4.0 * s2) + 2.0 * np.exp(3.0 * s2)
                   + 3.0 * np.exp(2.0 * s2) - 6.0)
    else:
        skew, ex_kurt = 0.0, 0.0
    return {"mean": float(mean), "variance": float(var),
            "skewness": float(skew), "excess_kurtosis": float(ex_kurt)}


# ---------------------------------------------------------------------------
# Empirical marginal: running product of exp(X_t) with X_t ~ N(0,1) iid.
# ---------------------------------------------------------------------------

def empirical_log_normal(n: int, n_reps: int,
                         rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """Return (S_n samples, log S_n samples).

    Computed via the identity log(prod M_t) = sum(X_t), which is both
    the natural mathematical statement and the numerically stable way to
    sample exp(W) for W = N(0, n) when n is large.
    """
    W = rng.standard_normal((n_reps, n)).sum(axis=1)
    S = np.exp(W)
    return S, W


def histogram_vs_density(s_n: np.ndarray, sigma_squared: float,
                          bins: int = 80) -> dict:
    # Use log-spaced bins for the heavy right tail.
    if len(s_n) == 0 or s_n.max() <= 0:
        return None
    log_lo = float(np.log(max(s_n.min(), 1e-12)))
    log_hi = float(np.log(s_n.max()))
    edges = np.exp(np.linspace(log_lo, log_hi, bins + 1))
    counts, _ = np.histogram(s_n, bins=edges, density=True)
    centres = np.sqrt(edges[:-1] * edges[1:])
    pdf = lognormal_pdf(centres, sigma_squared)
    widths = np.diff(edges)
    return {"centres": centres, "hist_density": counts,
            "pdf_analytical": pdf, "widths": widths}


def l1_density_distance(h: dict) -> float:
    return float(np.sum(np.abs(h["hist_density"] - h["pdf_analytical"])
                        * h["widths"]))


def standardised_skew_kurt(s: np.ndarray) -> tuple[float, float]:
    mu = float(s.mean())
    sd = float(s.std())
    if sd <= 0:
        return 0.0, 0.0
    skew = float(((s - mu) ** 3).mean() / sd ** 3)
    ex_kurt = float(((s - mu) ** 4).mean() / sd ** 4) - 3.0
    return skew, ex_kurt


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def main():
    rng = np.random.default_rng(0)

    horizons = [1, 2, 10, 100]
    n_reps = 200_000

    print("Step 1.  Both sides at each horizon: multiplicative S_n and additive log S_n.")
    marginal_rows = []
    two_sides_rows = []
    for n in horizons:
        S, W = empirical_log_normal(n, n_reps, rng)
        # multiplicative side moments
        S_mean = float(S.mean())
        S_var = float(S.var())
        S_skew, S_ek = standardised_skew_kurt(S)
        ln_moms = lognormal_moments(n)
        # additive side moments (log)
        W_mean = float(W.mean())
        W_var = float(W.var())
        W_skew, W_ek = standardised_skew_kurt(W)
        # density comparison on the multiplicative side
        h = histogram_vs_density(S, n)
        l1 = l1_density_distance(h) if h is not None else float("nan")

        print(f"  n = {n:>4d}")
        print(f"    multiplicative: mean = {S_mean:>10.2f} (pred "
              f"{ln_moms['mean']:>10.4f})   var = {S_var:>10.2f} (pred "
              f"{ln_moms['variance']:>14.4f})")
        print(f"                    skew (emp) = {S_skew:>8.2f} (pred "
              f"{ln_moms['skewness']:>10.2f})   excess kurt (emp) = "
              f"{S_ek:>10.2f}")
        print(f"    additive (log): mean = {W_mean:+.4f}   var = "
              f"{W_var:.4f} (pred {n})   skew = {W_skew:+.4f}   excess kurt = "
              f"{W_ek:+.4f}")
        print(f"    L¹ (hist vs density) = {l1:.4f}")

        two_sides_rows.append({
            "horizon": n,
            "mult_mean": S_mean, "mult_mean_predicted": ln_moms["mean"],
            "mult_variance": S_var, "mult_variance_predicted": ln_moms["variance"],
            "mult_skew_empirical": S_skew, "mult_skew_predicted": ln_moms["skewness"],
            "mult_excess_kurt_empirical": S_ek,
            "mult_excess_kurt_predicted": ln_moms["excess_kurtosis"],
            "log_mean": W_mean, "log_mean_predicted": 0.0,
            "log_variance": W_var, "log_variance_predicted": float(n),
            "log_skew": W_skew, "log_skew_predicted": 0.0,
            "log_excess_kurt": W_ek, "log_excess_kurt_predicted": 0.0,
            "l1_density_distance_mult_side": l1,
        })
        if h is not None:
            for i in range(len(h["centres"])):
                marginal_rows.append({
                    "horizon": n,
                    "x": float(h["centres"][i]),
                    "empirical_density": float(h["hist_density"][i]),
                    "analytical_lognormal_density": float(h["pdf_analytical"][i]),
                    "abs_diff": float(abs(h["hist_density"][i] -
                                           h["pdf_analytical"][i])),
                })

    # --- CSVs ------------------------------------------------------------
    with (RESULTS / "E09_marginal_horizons.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["horizon", "x",
                                          "empirical_density",
                                          "analytical_lognormal_density",
                                          "abs_diff"])
        w.writeheader()
        for r in marginal_rows:
            w.writerow(r)

    with (RESULTS / "E09_two_sides.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(two_sides_rows[0].keys()))
        w.writeheader()
        for r in two_sides_rows:
            w.writerow(r)

    # --- Readout ---------------------------------------------------------
    md = [
        "# E09 — The log-normal trajectory (running product)",
        "",
        "**Step rule:** M_t = exp(X_t) with X_t ~ N(0, 1) independent across t.  "
        "**Trajectory (multiplicative side):** S_t = M_1 · M_2 · … · M_t.  "
        "**Trajectory (additive side):** W_t = log S_t = X_1 + X_2 + … + X_t  "
        "(the Gaussian walk of E06).  "
        "**Analytical marginal at horizon n:** S_n has the log-normal "
        "density with parameters (μ = 0, σ² = n); W_n has the Gaussian "
        "density N(0, n).",
        "",
        "## What is new in this experiment",
        "",
        "All earlier experiments accumulated steps additively. Here the "
        "accumulation is multiplicative — the state at step t is a "
        "product, not a sum. This is the first qualitative change to the "
        "accumulation machinery in the programme. The change is, however, "
        "less mysterious than it sounds: the logarithm of the trajectory "
        "is exactly the additive Gaussian walk we already know from E06. "
        "So everything that happens here is the Gaussian walk seen through "
        "an exponential lens, and the new ingredients show up on the "
        "*multiplicative* side as an unbounded amplification of moments.",
        "",
        "## The two sides at each horizon",
        "",
        "Reading both the multiplicative S_n and the additive W_n = log S_n "
        "from the same trajectories:",
        "",
        "**Multiplicative side** — the moments of S_n grow with n; the mean "
        "grows as exp(n/2), the variance grows as exp(2n) − exp(n), the "
        "skewness as (e^n + 2)·√(e^n − 1).",
        "",
        "| horizon n | mean (emp) | mean (pred) | var (emp) | var (pred) | skew (emp) | skew (pred) |",
        "| ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for r in two_sides_rows:
        md.append(f"| {r['horizon']} | "
                  f"{r['mult_mean']:.2f} | "
                  f"{r['mult_mean_predicted']:.4f} | "
                  f"{r['mult_variance']:.2e} | "
                  f"{r['mult_variance_predicted']:.4e} | "
                  f"{r['mult_skew_empirical']:.2f} | "
                  f"{r['mult_skew_predicted']:.2e} |")

    md += [
        "",
        "Even at moderate horizons (n = 100), the predicted skewness of "
        "the log-normal marginal is astronomical. The empirical estimate "
        "from 200 000 samples does not match these predictions for large n: "
        "the tail of the log-normal at large σ² is so extreme that any "
        "finite sample under-represents it, and the empirical skewness is "
        "dominated by the largest few samples. **This is itself the "
        "qualitative finding:** on the multiplicative side, the shape-decay "
        "rule from note 02 does *not* apply — every horizon brings more "
        "extreme tails, not less.",
        "",
        "**Additive side** — taking the logarithm of the trajectory "
        "recovers the Gaussian walk of E06: the variance of W_n grows "
        "linearly with n, skew and excess kurtosis stay near zero, and the "
        "shape-decay rule applies in its trivial form (the step had no "
        "shape to decay to begin with).",
        "",
        "| horizon n | mean (emp) | mean (pred) | var (emp) | var (pred) | skew (emp) | excess kurt (emp) |",
        "| ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for r in two_sides_rows:
        md.append(f"| {r['horizon']} | "
                  f"{r['log_mean']:+.4f} | "
                  f"{r['log_mean_predicted']} | "
                  f"{r['log_variance']:.4f} | "
                  f"{r['log_variance_predicted']:.0f} | "
                  f"{r['log_skew']:+.4f} | "
                  f"{r['log_excess_kurt']:+.4f} |")

    md += [
        "",
        "Perfectly tame, as expected — the additive side is E06 by another "
        "name.",
        "",
        "## Empirical histogram vs analytical log-normal density",
        "",
        "On the multiplicative side, we compare the histogram of S_n (in "
        "log-spaced bins, to handle the heavy right tail) against the "
        "log-normal density with parameters (0, n). The L¹ distance between "
        "the two normalised densities:",
        "",
        "| horizon n | L¹ (hist vs density, multiplicative side) |",
        "| ---: | ---: |",
    ]
    for r in two_sides_rows:
        md.append(f"| {r['horizon']} | "
                  f"{r['l1_density_distance_mult_side']:.4f} |")

    md += [
        "",
        "The histogram tracks the density to sampling-noise level even "
        "though the moments diverge from their predicted values for large "
        "n. The *shape* of the marginal is correct; the moment-based "
        "summary just fails to describe it once the tail is heavy enough "
        "that finite samples cannot resolve it.",
        "",
        "## The substrate's view of this trajectory",
        "",
        "Two equivalent readings, depending on which side we look from:",
        "",
        "  - **Additive side:** the operator is the cumulative identity, "
        "the step is a Gaussian, the marginal at horizon n is N(0, n). "
        "Same as E06 exactly.",
        "",
        "  - **Multiplicative side:** the operator multiplies by the next "
        "M_t. Reading at the log scale linearises this to the additive "
        "operator above. The marginal at horizon n on this side is "
        "log-normal, and its moments grow with n rather than shrink — but "
        "the *shape* in log-space is exactly the Gaussian walk and follows "
        "the additive rules.",
        "",
        "## What this experiment shows",
        "",
        "When the accumulation is multiplicative, the underlying additive "
        "structure lives in the logarithm. The classical log-normal "
        "distribution at horizon n is exp(W_n) where W_n is the Gaussian "
        "walk after n steps. So far the convolution identity has been the "
        "single source of every marginal we computed; here we see its "
        "first generalisation: when the operation is product rather than "
        "sum, the same identity applies on the log scale.",
        "",
        "The shape-decay rule from note 02 still controls the additive "
        "side. On the multiplicative side, the corresponding statement is "
        "a *growth* of shape — moments of S_n inflate with n at "
        "exponential rates, not because the rule fails but because "
        "the rule is the wrong invariant for this operation. The right "
        "invariant lives in the logarithm.",
        "",
    ]
    (RESULTS / "E09_log_normal_product.md").write_text("\n".join(md))
    print(f"\nWrote {RESULTS / 'E09_log_normal_product.md'}")


if __name__ == "__main__":
    main()
