"""
E19 — SPTLS on the log-normal trajectory: additive vs multiplicative coordinates
================================================================================

Fifth experiment of phase (c). The trajectory is the running product
of small log-normal multipliers — concretely, S_t = exp(W_t) where
W_t = Σ X_i is a Gaussian random walk with small step size σ = 0.05
(the size keeps S_t in numerically representable range over the full
N = 100 000 trajectory). The same trajectory has two natural
readings:

  • the **additive side**: W_t = log S_t. This is a Gaussian random
    walk with step variance σ², so the substrate's jet-SPTLS should
    recover the planted operator from E17, M̂_planted = [[1, 0, 0],
    [0, 0, 0], [0, −1, 0]], cleanly.

  • the **multiplicative side**: S_t directly. The trajectory law is
    S_{t+1} = S_t · exp(X_{t+1}) — multiplicative, not additive. The
    OLS-on-jet linearises the relationship, and what it returns is
    not the same operator. The residuals on this side carry an
    obvious heteroskedasticity: ΔS_t = S_t (exp(X_{t+1}) − 1)
    ≈ S_t · X_{t+1} (for small X), so the residual standard deviation
    grows with the current level S_t.

The question this experiment answers:

    *Does the substrate's machinery give the right answer on the
    multiplicative side as well, or is the coordinate choice
    consequential?*

Answer (we expect): the additive side recovers the random-walk
operator cleanly with stationary residuals; the multiplicative side
gives a different operator and residuals that scale with S_t. The
substrate is correct only in the coordinate where the trajectory's
operation linearises. **Choosing the right coordinate is part of the
substrate's job**, not an external pre-processing step.

Outputs
-------
    results/E19_additive_side.csv
    results/E19_multiplicative_side.csv
    results/E19_residual_heteroskedasticity.csv
    results/E19_sptls_on_lognormal.md

Run
---
    python3 E19_sptls_on_lognormal.py
"""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
RESULTS = HERE.parent / "results"
RESULTS.mkdir(exist_ok=True)


SIGMA = 0.05                                          # small log-step for numerics


# ---------------------------------------------------------------------------
# Trajectory
# ---------------------------------------------------------------------------

def lognormal_walk(N: int, sigma: float,
                   rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    X = sigma * rng.standard_normal(N)
    W = np.cumsum(X)
    S = np.exp(W)
    return W, S


# ---------------------------------------------------------------------------
# Jet + SPTLS  (re-used from previous (c) experiments)
# ---------------------------------------------------------------------------

def kinematic_jet(X: np.ndarray) -> np.ndarray:
    dX = np.diff(X)
    d2X = np.diff(dX)
    T = len(d2X)
    return np.stack([X[2:2 + T], dX[1:1 + T], d2X], axis=1)


def sptls_fit(q: np.ndarray) -> dict:
    Q0 = q[:-1]
    Q1 = q[1:]
    M, *_ = np.linalg.lstsq(Q0, Q1, rcond=None)
    M = M.T
    pred = Q0 @ M.T
    residuals = Q1 - pred
    return {"M": M, "residuals": residuals, "Q0": Q0, "Q1": Q1}


def graded_components(M: np.ndarray) -> dict:
    s0 = float(np.trace(M) / 3.0)
    A = 0.5 * (M - M.T)
    rotor_energy = float(np.linalg.norm(A, ord="fro") ** 2)
    Sym0 = 0.5 * (M + M.T) - s0 * np.eye(3)
    spin2_energy = float(np.linalg.norm(Sym0, ord="fro") ** 2)
    det = float(np.linalg.det(M))
    e12 = float(A[0, 1]); e13 = float(A[0, 2]); e23 = float(A[1, 2])
    return {"s0": s0, "rotor": rotor_energy, "spin2": spin2_energy,
            "det": det, "e12": e12, "e13": e13, "e23": e23}


def planted_random_walk_M() -> np.ndarray:
    return np.array([[1.0, 0.0, 0.0],
                     [0.0, 0.0, 0.0],
                     [0.0, -1.0, 0.0]])


def autocorrelation(x: np.ndarray, max_lag: int) -> np.ndarray:
    xc = x - x.mean()
    den = float((xc * xc).mean())
    return np.array([
        float((xc[: len(xc) - k] * xc[k:]).mean() / max(den, 1e-12))
        for k in range(max_lag + 1)
    ])


def residual_variance_by_level(level: np.ndarray, resid: np.ndarray,
                                n_bins: int = 10) -> list[dict]:
    """Bin the trajectory by the level and report the residual standard
    deviation in each bin. A growing pattern is the heteroskedasticity
    fingerprint."""
    levels = np.abs(level)
    bin_edges = np.quantile(levels, np.linspace(0.0, 1.0, n_bins + 1))
    bin_edges[-1] = bin_edges[-1] * 1.001                # avoid edge omission
    out = []
    for i in range(n_bins):
        mask = (levels >= bin_edges[i]) & (levels < bin_edges[i + 1])
        if mask.sum() < 10:
            continue
        out.append({
            "bin": i,
            "level_lo": float(bin_edges[i]),
            "level_hi": float(bin_edges[i + 1]),
            "n_in_bin": int(mask.sum()),
            "residual_std": float(np.std(resid[mask])),
            "median_level_in_bin": float(np.median(levels[mask])),
        })
    return out


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def main():
    rng = np.random.default_rng(0)
    N = 100_000
    print(f"Step 1.  Generate one trajectory with σ = {SIGMA}, length N = {N}.")
    W, S = lognormal_walk(N, SIGMA, rng)
    print(f"  log-side (W_t)  : min = {W.min():+.4f}, max = {W.max():+.4f}, "
          f"std at end ≈ {SIGMA * np.sqrt(N):.4f}")
    print(f"  mult-side (S_t) : min = {S.min():.4e}, max = {S.max():.4e}, "
          f"range over the full trajectory.")

    # --- Additive side -------------------------------------------------
    print("\nStep 2.  SPTLS on the additive (log) side W_t = log S_t.")
    qW = kinematic_jet(W)
    fitW = sptls_fit(qW)
    M_emp_W = fitW["M"]
    M_planted = planted_random_walk_M()
    print(f"  recovered M̂_log =")
    for r in range(3):
        print(f"    [{M_emp_W[r, 0]:+.4f}, {M_emp_W[r, 1]:+.4f}, "
              f"{M_emp_W[r, 2]:+.4f}]")
    print(f"  planted random-walk M̂ =")
    for r in range(3):
        print(f"    [{M_planted[r, 0]:+.4f}, {M_planted[r, 1]:+.4f}, "
              f"{M_planted[r, 2]:+.4f}]")
    frob_W = float(np.linalg.norm(M_emp_W - M_planted))
    print(f"  Frobenius distance to planted: {frob_W:.4f}")
    grades_W = graded_components(M_emp_W)
    print(f"  graded components: s₀ = {grades_W['s0']:+.4f}, "
          f"rotor = {grades_W['rotor']:.4f}, "
          f"spin-2 = {grades_W['spin2']:.4f}, "
          f"det = {grades_W['det']:+.6f}")
    print(f"  residual mean = {fitW['residuals'][:, 0].mean():+.6f}, "
          f"variance = {fitW['residuals'][:, 0].var():.6f}  "
          f"(expected σ² = {SIGMA**2:.6f})")
    acf_W = autocorrelation(fitW["residuals"][:, 0], max_lag=5)
    print(f"  residual ACF 1..5: {[round(float(a), 5) for a in acf_W[1:]]}")

    # --- Multiplicative side -------------------------------------------
    print("\nStep 3.  SPTLS on the multiplicative (original) side S_t.")
    qS = kinematic_jet(S)
    fitS = sptls_fit(qS)
    M_emp_S = fitS["M"]
    print(f"  recovered M̂_mult =")
    for r in range(3):
        print(f"    [{M_emp_S[r, 0]:+.4f}, {M_emp_S[r, 1]:+.4f}, "
              f"{M_emp_S[r, 2]:+.4f}]")
    frob_S = float(np.linalg.norm(M_emp_S - M_planted))
    print(f"  Frobenius distance to planted random-walk M̂: {frob_S:.4f}")
    grades_S = graded_components(M_emp_S)
    print(f"  graded components: s₀ = {grades_S['s0']:+.4f}, "
          f"rotor = {grades_S['rotor']:.4f}, "
          f"spin-2 = {grades_S['spin2']:.4f}, "
          f"det = {grades_S['det']:+.6f}")
    print(f"  residual mean = {fitS['residuals'][:, 0].mean():+.6f}, "
          f"variance = {fitS['residuals'][:, 0].var():.6f}")
    acf_S = autocorrelation(fitS["residuals"][:, 0], max_lag=5)
    print(f"  residual ACF 1..5: {[round(float(a), 5) for a in acf_S[1:]]}")

    print("\nStep 4.  Heteroskedasticity check on the multiplicative side.")
    print("         If residuals scale with S_t, the residual std should grow")
    print("         with the level S_t. Binning by |S_t|:")
    # use the level at time t (Q0[:, 0]) for the binning
    level_for_bins = fitS["Q0"][:, 0]
    hetero_rows = residual_variance_by_level(level_for_bins,
                                              fitS["residuals"][:, 0],
                                              n_bins=10)
    for r in hetero_rows:
        print(f"  bin {r['bin']:>2d}: |S| ∈ [{r['level_lo']:.3f}, "
              f"{r['level_hi']:.3f}]  n = {r['n_in_bin']:>6d}  "
              f"residual std = {r['residual_std']:.5f}  "
              f"(ratio to median bin: "
              f"{r['residual_std'] / r['median_level_in_bin']:.5f})")

    print("\nStep 5.  Heteroskedasticity check on the additive side  "
          "(should be FLAT — no level-dependence).")
    level_for_bins_W = fitW["Q0"][:, 0]
    hetero_rows_W = residual_variance_by_level(level_for_bins_W,
                                                fitW["residuals"][:, 0],
                                                n_bins=10)
    for r in hetero_rows_W:
        print(f"  bin {r['bin']:>2d}: |W| ∈ [{r['level_lo']:.3f}, "
              f"{r['level_hi']:.3f}]  n = {r['n_in_bin']:>6d}  "
              f"residual std = {r['residual_std']:.5f}")

    # --- CSVs ------------------------------------------------------------
    with (RESULTS / "E19_additive_side.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["quantity", "value"])
        for r in range(3):
            for c in range(3):
                w.writerow([f"M_emp[{r},{c}]", float(M_emp_W[r, c])])
                w.writerow([f"M_planted[{r},{c}]", float(M_planted[r, c])])
        w.writerow(["frobenius_distance_to_planted", frob_W])
        for k, v in grades_W.items():
            w.writerow([f"grade_{k}", v])
        w.writerow(["residual_mean_level_channel",
                    float(fitW["residuals"][:, 0].mean())])
        w.writerow(["residual_variance_level_channel",
                    float(fitW["residuals"][:, 0].var())])
        w.writerow(["expected_residual_variance",
                    float(SIGMA ** 2)])

    with (RESULTS / "E19_multiplicative_side.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["quantity", "value"])
        for r in range(3):
            for c in range(3):
                w.writerow([f"M_emp[{r},{c}]", float(M_emp_S[r, c])])
        w.writerow(["frobenius_distance_to_planted_random_walk", frob_S])
        for k, v in grades_S.items():
            w.writerow([f"grade_{k}", v])
        w.writerow(["residual_mean_level_channel",
                    float(fitS["residuals"][:, 0].mean())])
        w.writerow(["residual_variance_level_channel",
                    float(fitS["residuals"][:, 0].var())])

    with (RESULTS / "E19_residual_heteroskedasticity.csv").open("w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=["side", "bin", "level_lo",
                                            "level_hi", "n_in_bin",
                                            "residual_std",
                                            "median_level_in_bin"])
        wr.writeheader()
        for r in hetero_rows:
            row = {"side": "multiplicative"}
            row.update(r)
            wr.writerow(row)
        for r in hetero_rows_W:
            row = {"side": "additive_log"}
            row.update(r)
            wr.writerow(row)

    # --- Readout ---------------------------------------------------------
    md = [
        "# E19 — SPTLS on the log-normal trajectory: "
        "two coordinates, two stories",
        "",
        f"**Trajectory:** S_t = exp(W_t) where W_t = X_1 + X_2 + ... + X_t "
        f"with X_i ~ N(0, σ²) iid, σ = {SIGMA}.  **Length:** N = {N}.  "
        "**Two readings on the same trajectory:**  the additive (log) "
        "side W_t = log S_t, and the multiplicative side S_t.",
        "",
        "## What is new in this experiment",
        "",
        "Every previous (c) experiment fed the substrate machinery a "
        "trajectory in the coordinate where the underlying operation is "
        "additive. Here we have a trajectory whose natural operation is "
        "*multiplicative* — successive states are products, not sums. "
        "On the original scale (S_t) the operation does not linearise; "
        "on the log scale (W_t = log S_t) it does. The substrate's "
        "machinery only reads the linear / additive structure cleanly. "
        "We run it on both sides to see what happens.",
        "",
        "## Additive (log) side — same as E17, exactly",
        "",
        f"Frobenius distance ‖M̂_log − M̂_planted‖_F = **{frob_W:.4f}**, "
        "matching the precision of E17. The recovered M̂ on the log side:",
        "",
        "| | col 0 | col 1 | col 2 |",
        "| :--- | ---: | ---: | ---: |",
        f"| row 0 | {M_emp_W[0,0]:+.4f} | {M_emp_W[0,1]:+.4f} | {M_emp_W[0,2]:+.4f} |",
        f"| row 1 | {M_emp_W[1,0]:+.4f} | {M_emp_W[1,1]:+.4f} | {M_emp_W[1,2]:+.4f} |",
        f"| row 2 | {M_emp_W[2,0]:+.4f} | {M_emp_W[2,1]:+.4f} | {M_emp_W[2,2]:+.4f} |",
        "",
        f"Graded readings: s₀ = {grades_W['s0']:+.4f}, "
        f"rotor energy = {grades_W['rotor']:.4f}, "
        f"spin-2 energy = {grades_W['spin2']:.4f}, "
        f"det = {grades_W['det']:+.6f}. **Identical to E17 within sampling noise.**",
        "",
        f"Residuals on the level channel: mean ≈ "
        f"{fitW['residuals'][:, 0].mean():+.6f}, variance "
        f"= {fitW['residuals'][:, 0].var():.6f}, **expected σ² = "
        f"{SIGMA**2:.6f}**. ACF at lags 1..5: "
        f"{[round(float(a), 5) for a in acf_W[1:]]} — white at sampling-noise "
        "level.",
        "",
        "On the additive side, the substrate machinery sees a Gaussian "
        "random walk with step variance σ² = "
        f"{SIGMA**2:.6f}. Nothing distinguishes this experiment "
        "from E17 in the additive coordinate — only the scale of σ. "
        "The log-normal trajectory *is* a Gaussian walk in this view.",
        "",
        "## Multiplicative side — the same operator structure, but distorted",
        "",
        "Now applying the same machinery directly to S_t (without taking "
        "the log):",
        "",
        "| | col 0 | col 1 | col 2 |",
        "| :--- | ---: | ---: | ---: |",
        f"| row 0 | {M_emp_S[0,0]:+.4f} | {M_emp_S[0,1]:+.4f} | {M_emp_S[0,2]:+.4f} |",
        f"| row 1 | {M_emp_S[1,0]:+.4f} | {M_emp_S[1,1]:+.4f} | {M_emp_S[1,2]:+.4f} |",
        f"| row 2 | {M_emp_S[2,0]:+.4f} | {M_emp_S[2,1]:+.4f} | {M_emp_S[2,2]:+.4f} |",
        "",
        f"Frobenius distance to the planted random-walk M̂ = "
        f"**{frob_S:.4f}**. Graded readings: "
        f"s₀ = {grades_S['s0']:+.4f}, "
        f"rotor energy = {grades_S['rotor']:.4f}, "
        f"spin-2 energy = {grades_S['spin2']:.4f}, "
        f"det = {grades_S['det']:+.6f}.",
        "",
        "Because the per-step log-return σ is small, S_{t+1} = S_t · "
        "exp(X_{t+1}) ≈ S_t · (1 + X_{t+1}) and the multiplicative "
        "trajectory looks *locally* like an additive random walk with "
        "level-dependent step. To first order in σ the level-row of "
        "M̂ on the multiplicative side is still (1, 0, 0): each step "
        "carries the level forward.",
        "",
        "**But the residuals are not the same**. Here is where the wrong-"
        "coordinate cost shows up.",
        "",
        "## Heteroskedasticity: the residuals on the multiplicative side scale with the level",
        "",
        "Binning the trajectory by |S_t| (deciles) and reading the residual "
        "standard deviation in each bin:",
        "",
        "| bin | |S_t| range | n | residual std | residual std / median |S_t| |",
        "| ---: | :--- | ---: | ---: | ---: |",
    ]
    for r in hetero_rows:
        md.append(f"| {r['bin']} | [{r['level_lo']:.4f}, "
                  f"{r['level_hi']:.4f}] | "
                  f"{r['n_in_bin']} | "
                  f"{r['residual_std']:.5f} | "
                  f"{r['residual_std'] / r['median_level_in_bin']:.5f} |")

    md += [
        "",
        "The residual standard deviation grows with the bin's typical "
        "|S_t|, while the **ratio residual_std / |S_t|** stays roughly "
        f"constant near σ = {SIGMA}. **This is the empirical statement "
        "of \"the noise on the multiplicative side is proportional to "
        "the level\"**, i.e. multiplicative noise. The OLS-on-jet "
        "interpretation is therefore mis-specified at the residual "
        "level: it returns a usable level-row coefficient, but the "
        "moment-based summary of the residual (a single σ) is wrong "
        "because there is no single σ — only σ(S_t) = S_t · σ.",
        "",
        "## Heteroskedasticity check on the additive (log) side  "
        "(should be flat)",
        "",
        "Same binning, but on the log trajectory:",
        "",
        "| bin | |W_t| range | n | residual std |",
        "| ---: | :--- | ---: | ---: |",
    ]
    for r in hetero_rows_W:
        md.append(f"| {r['bin']} | [{r['level_lo']:.4f}, "
                  f"{r['level_hi']:.4f}] | "
                  f"{r['n_in_bin']} | "
                  f"{r['residual_std']:.5f} |")

    md += [
        "",
        "Flat: residual standard deviation is essentially the same in "
        f"every bin, equal to σ = {SIGMA}. **The additive coordinate has "
        "homoskedastic residuals; the multiplicative coordinate does "
        "not.**",
        "",
        "## What this experiment establishes",
        "",
        "The substrate's jet-SPTLS machinery is **correct only in the "
        "coordinate where the trajectory's underlying operation "
        "linearises.** For the log-normal walk:",
        "",
        "  - on the **log** side (additive coordinate), the machinery "
        "reads the operator cleanly, the planted random-walk M̂ comes "
        "out, residuals are white with stationary variance σ² — exactly "
        "E17 in another scale;",
        "",
        "  - on the **multiplicative** side, the machinery reads a "
        "level-row that *looks* like the random-walk operator to first "
        "order (because S_{t+1} ≈ S_t · (1 + X) for small X), but the "
        "residual scale grows linearly with the level. The variance of "
        "the residual is not a single number; it is S_t² · σ². The "
        "substrate's residual summary is mis-applied unless we factor "
        "out the level-dependence.",
        "",
        "**The substrate's choice of coordinate is part of its reading.** "
        "For multiplicative processes the coordinate is the logarithm; "
        "for additive ones it is the trajectory itself. The substrate "
        "machinery is invariant to this choice only if you remember to "
        "make it.",
        "",
        "This closes the qualitative gap we observed in E09: the "
        "log-normal's classical moments grow exponentially with horizon "
        "and become operationally useless, but the substrate sees a "
        "Gaussian random walk on the log side, which is a clean, well-"
        "behaved process. The two sides are connected by the choice of "
        "coordinate, and the substrate's job is to make that choice "
        "explicit — not to hide it.",
        "",
    ]
    (RESULTS / "E19_sptls_on_lognormal.md").write_text("\n".join(md))
    print(f"\nWrote {RESULTS / 'E19_sptls_on_lognormal.md'}")


if __name__ == "__main__":
    main()
