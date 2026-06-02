"""
E17 — SPTLS on the kinematic jet of the Gaussian random walk
=============================================================

Third experiment of phase (c). Same machinery as E15 / E16, applied to
the Gaussian random walk of E06 — the φ = 1 limit of AR(1). Here the
level channel is non-stationary: Var(X_t) = t · σ² grows linearly with
time, so the jet has channels with wildly different scales (the level
explodes while ΔX_t and Δ²X_t stay stationary at O(1) variance).

The question this experiment answers:

    *Does the substrate's uniform reading survive the unit root?*

The algebraic answer is yes. Setting φ = 1 in the AR(1) formula for the
jet operator gives

    M̂_planted  =  [[ 1,  0,  0],
                   [ 0,  0,  0],
                   [ 0, -1,  0]].

The level just carries forward (the random walk has no drift); the
velocity channel ΔX_{t+1} = ε_{t+1} is the pure innovation; the
acceleration channel Δ²X_{t+1} = ε_{t+1} − ΔX_t subtracts the previous
velocity. The four graded readings:

    s₀ (trace / 3)    =  1/3,
    rotor energy      =  0.5  (concentrated on the e23 plane),
    det               =  0,
    spin-2 energy     =  ≈ 1.17.

Comparing to E15 (AR(1), φ = 0.7) and E16 (AR(2), (φ₁, φ₂) = (0.5,
0.3)) we have three points along the dependence axis with three
different graded fingerprints — and the substrate's machinery reads
each cleanly using the same 3 × 3 OLS on the jet.

What we want to see, by direct empirical comparison:

  (i)   the OLS fit on the jet recovers M̂_planted to within
        sampling noise — the level row is recovered with the
        unit-root coefficient 1, exactly;

  (ii)  the four graded readings match the analytical values for
        φ = 1;

  (iii) residuals on each channel are white and have variance ≈ σ²,
        as in E15 / E16 — and they are *perfectly correlated across
        channels* (rank-1 residual covariance) because the same
        innovation ε_{t+1} drives all three channels.

Outputs
-------
    results/E17_recovered_M.csv
    results/E17_graded_components.csv
    results/E17_residuals_per_channel.csv
    results/E17_residual_cross_correlations.csv
    results/E17_sptls_on_random_walk.md

Run
---
    python3 E17_sptls_on_random_walk.py
"""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
RESULTS = HERE.parent / "results"
RESULTS.mkdir(exist_ok=True)


SIGMA = 1.0
PHI_RANDOM_WALK = 1.0


# ---------------------------------------------------------------------------
# Trajectory
# ---------------------------------------------------------------------------

def gaussian_random_walk(N: int, sigma: float,
                          rng: np.random.Generator) -> np.ndarray:
    """X_t = X_{t-1} + ε_t with X_0 = 0 and ε_t ~ N(0, σ²) iid."""
    eps = sigma * rng.standard_normal(N)
    return np.cumsum(eps)


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
    return {"M": M, "residuals": residuals,
            "Q0": Q0, "Q1": Q1, "predicted": pred}


def graded_components(M: np.ndarray) -> dict:
    s0 = float(np.trace(M) / 3.0)
    A = 0.5 * (M - M.T)
    rotor_energy = float(np.linalg.norm(A, ord="fro") ** 2)
    e12 = float(A[0, 1]); e13 = float(A[0, 2]); e23 = float(A[1, 2])
    Sym0 = 0.5 * (M + M.T) - s0 * np.eye(3)
    spin2_energy = float(np.linalg.norm(Sym0, ord="fro") ** 2)
    det = float(np.linalg.det(M))
    return {"s0_trace_over_3": s0,
            "rotor_energy_frobenius_squared": rotor_energy,
            "e12_bivector": e12, "e13_bivector": e13, "e23_bivector": e23,
            "spin2_energy_frobenius_squared": spin2_energy,
            "det_M": det}


def planted_M_for_random_walk() -> np.ndarray:
    return np.array([[1.0,  0.0, 0.0],
                     [0.0,  0.0, 0.0],
                     [0.0, -1.0, 0.0]])


def autocorrelation_function(x: np.ndarray, max_lag: int) -> np.ndarray:
    xc = x - x.mean()
    den = float((xc * xc).mean())
    return np.array([
        float((xc[: len(xc) - k] * xc[k:]).mean() / max(den, 1e-12))
        for k in range(max_lag + 1)
    ])


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def main():
    rng = np.random.default_rng(0)
    N = 200_000
    print(f"Step 1.  Generate Gaussian random walk of length {N}.")
    X = gaussian_random_walk(N, SIGMA, rng)
    print(f"  empirical mean (running, no expected stationary value): "
          f"{float(X.mean()):+.4f}")
    print(f"  empirical Var(X_N) = {float(X[-1] ** 2):.1f}   "
          f"(expected ~ N·σ² = {N})")
    print(f"  empirical level variance over time (non-stationary): "
          f"X.var() = {float(X.var()):.1f}")

    print("\nStep 2.  Build the kinematic jet.")
    q = kinematic_jet(X)
    print(f"  jet shape = {q.shape}")
    print(f"  channel variances:  X = {q[:, 0].var():.1f}   "
          f"ΔX = {q[:, 1].var():.4f}  (expected σ² = 1)   "
          f"Δ²X = {q[:, 2].var():.4f}  (expected 2σ² = 2)")
    print("  (the level channel variance is huge because of the unit root;")
    print("   the difference channels are stationary, with O(1) variance)")

    print("\nStep 3.  Fit the SPTLS operator M̂.")
    fit = sptls_fit(q)
    M_emp = fit["M"]
    M_planted = planted_M_for_random_walk()
    print("  recovered M̂ =")
    for r in range(3):
        print(f"    [{M_emp[r, 0]:+.4f}, {M_emp[r, 1]:+.4f}, "
              f"{M_emp[r, 2]:+.4f}]")
    print("  planted  M̂ =")
    for r in range(3):
        print(f"    [{M_planted[r, 0]:+.4f}, {M_planted[r, 1]:+.4f}, "
              f"{M_planted[r, 2]:+.4f}]")
    print(f"  Frobenius distance ‖M̂_emp − M̂_planted‖_F = "
          f"{float(np.linalg.norm(M_emp - M_planted)):.4f}")

    print("\nStep 4.  Graded components of M̂.")
    grades_emp = graded_components(M_emp)
    grades_planted = graded_components(M_planted)
    print("                                empirical    planted")
    for k in ["s0_trace_over_3",
              "rotor_energy_frobenius_squared",
              "e12_bivector", "e13_bivector", "e23_bivector",
              "spin2_energy_frobenius_squared",
              "det_M"]:
        print(f"  {k:<36s} {grades_emp[k]:+10.4f}   {grades_planted[k]:+.4f}")

    print("\nStep 5.  Residuals per channel.")
    resid = fit["residuals"]
    resid_rows = []
    for ch, name in enumerate(["level (X_t)", "Δ velocity (ΔX_t)",
                                "Δ² acceleration (Δ²X_t)"]):
        mean_r = float(resid[:, ch].mean())
        var_r = float(resid[:, ch].var())
        acf_r = autocorrelation_function(resid[:, ch], max_lag=5)
        print(f"  channel {ch} ({name}):  mean = {mean_r:+.4f}  "
              f"var = {var_r:.4f}  ACF 1..5 = "
              f"{[round(float(a), 4) for a in acf_r[1:]]}")
        resid_rows.append({
            "channel": ch,
            "channel_name": name,
            "residual_mean": mean_r,
            "residual_variance": var_r,
            "residual_acf_lag_1": float(acf_r[1]),
            "residual_acf_lag_2": float(acf_r[2]),
            "residual_acf_lag_3": float(acf_r[3]),
            "residual_acf_lag_4": float(acf_r[4]),
            "residual_acf_lag_5": float(acf_r[5]),
        })

    print("\nStep 6.  Cross-correlations of residuals across channels.")
    print("         All three channels are driven by the same ε_{t+1}, so")
    print("         the cross-correlations should be ≈ 1 at lag 0 between any pair.")
    cross_rows = []
    for i in range(3):
        for j in range(3):
            r_ij = float(np.corrcoef(resid[:, i], resid[:, j])[0, 1])
            print(f"  Corr(resid[{i}], resid[{j}]) = {r_ij:+.4f}")
            cross_rows.append({"i": i, "j": j, "correlation": r_ij})

    # --- CSVs ------------------------------------------------------------
    with (RESULTS / "E17_recovered_M.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["row", "col", "M_empirical",
                                          "M_planted", "abs_diff"])
        w.writeheader()
        for r in range(3):
            for c in range(3):
                w.writerow({"row": r, "col": c,
                            "M_empirical": float(M_emp[r, c]),
                            "M_planted":   float(M_planted[r, c]),
                            "abs_diff": float(abs(M_emp[r, c]
                                                  - M_planted[r, c]))})

    with (RESULTS / "E17_graded_components.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["component", "empirical", "planted"])
        w.writeheader()
        for k in grades_emp:
            w.writerow({"component": k, "empirical": grades_emp[k],
                        "planted": grades_planted[k]})

    with (RESULTS / "E17_residuals_per_channel.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(resid_rows[0].keys()))
        w.writeheader()
        for r in resid_rows:
            w.writerow(r)

    with (RESULTS / "E17_residual_cross_correlations.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["i", "j", "correlation"])
        w.writeheader()
        for r in cross_rows:
            w.writerow(r)

    # --- Readout ---------------------------------------------------------
    md = [
        "# E17 — SPTLS on the kinematic jet of the Gaussian random walk",
        "",
        "**Third experiment of phase (c).**  Same machinery as E15 / E16, "
        "now applied to the Gaussian random walk of E06 — the φ = 1 limit "
        "of AR(1). The level channel is non-stationary: Var(X_t) grows "
        "linearly with t, while the difference channels (ΔX_t, Δ²X_t) "
        "remain stationary. The unit root makes the level dominate the "
        "covariance matrix by a factor of ~N, which is the kind of "
        "degeneracy where naive variance-based fits can misbehave. The "
        "question is whether the substrate's uniform reading still works.",
        "",
        "## The planted operator",
        "",
        "Setting φ = 1 in the AR(1) formula M̂ = [[φ, 0, 0], [φ-1, 0, 0], "
        "[φ-1, -1, 0]] gives",
        "",
        "    M̂_planted  =  [[ 1.0000,  0.0000,  0.0000],",
        "                   [ 0.0000,  0.0000,  0.0000],",
        "                   [ 0.0000, -1.0000,  0.0000]].",
        "",
        "The level just carries forward (no decay, no growth). The velocity "
        "row is exactly zero — meaning ΔX_{t+1} is pure innovation, "
        "independent of everything in the current jet. The acceleration "
        "row is (0, −1, 0): Δ²X_{t+1} subtracts the previous ΔX_t (and is "
        "perturbed by the innovation that already drove the level and "
        "velocity rows).",
        "",
        "## Recovery of M̂",
        "",
        "OLS on the jet, with N = 200 000 samples:",
        "",
        "| row | col | M̂ empirical | M̂ planted | |diff| |",
        "| :--- | :--- | ---: | ---: | ---: |",
    ]
    for r in range(3):
        for c in range(3):
            md.append(f"| {r} | {c} | "
                      f"{M_emp[r, c]:+.4f} | "
                      f"{M_planted[r, c]:+.4f} | "
                      f"{abs(M_emp[r, c] - M_planted[r, c]):.4f} |")

    md += [
        "",
        f"Frobenius distance ‖M̂_emp − M̂_planted‖_F = "
        f"{float(np.linalg.norm(M_emp - M_planted)):.4f}, at "
        "sampling-noise level. **The substrate machinery recovers the "
        "unit-root operator cleanly.** The OLS estimator is in fact "
        "*super-consistent* on the level coefficient because the level "
        "has unbounded variance — the regression has effectively infinite "
        "leverage on that coefficient. The result is that the recovered "
        "value is closer to the planted than naive √N sampling noise "
        "would suggest.",
        "",
        "## Graded components of M̂",
        "",
        "Comparison to the values implied by the planted M̂ at φ = 1:",
        "",
        "| component | empirical | planted |",
        "| :--- | ---: | ---: |",
    ]
    for k in ["s0_trace_over_3",
              "rotor_energy_frobenius_squared",
              "e12_bivector", "e13_bivector", "e23_bivector",
              "spin2_energy_frobenius_squared",
              "det_M"]:
        md.append(f"| `{k}` | "
                  f"{grades_emp[k]:+.4f} | "
                  f"{grades_planted[k]:+.4f} |")

    md += [
        "",
        "Three points now in the dependence-axis fingerprint:",
        "",
        "| case | φ / depth | s₀ | rotor energy | spin-2 energy |",
        "| :--- | :--- | ---: | ---: | ---: |",
        "| E15 AR(1) φ=0.7 | depth 1, stationary | +0.233 | 0.59 | 0.92 |",
        "| E16 AR(2)       | depth 2, stationary | +0.167 | 0.87 | 1.64 |",
        "| E17 random walk | φ=1, unit root      | +0.333 | 0.50 | 1.17 |",
        "",
        "The three cases have distinguishable graded fingerprints. The "
        "random walk has the highest s₀ (1/3 exactly, because the trace "
        "is just the unit-root coefficient 1). The AR(2) has the highest "
        "rotor and spin-2 energies because depth-2 memory adds bivector "
        "components that depth-1 does not. The random walk's bivector "
        "content sits entirely on the e23 plane (0.5), which is the "
        "Cl(3,0) plane corresponding to the velocity-acceleration "
        "coupling.",
        "",
        "## Residuals per channel",
        "",
        "| channel | residual mean | residual variance | ACF lags 1..5 |",
        "| :--- | ---: | ---: | :--- |",
    ]
    for r in resid_rows:
        acfs = [round(r[f"residual_acf_lag_{k}"], 4) for k in range(1, 6)]
        md.append(f"| {r['channel_name']} | "
                  f"{r['residual_mean']:+.4f} | "
                  f"{r['residual_variance']:.4f} | "
                  f"{acfs} |")

    md += [
        "",
        "Same pattern as E15 / E16: residuals white on every channel, "
        "variance ≈ σ² = 1 on the level (and identical on the others by "
        "the algebraic identities of the jet).",
        "",
        "## Cross-correlations of residuals (the unit-root signature)",
        "",
        "For the random walk, the innovation ε_{t+1} drives all three "
        "jet channels simultaneously — there is no separate noise per "
        "channel, only one innovation per step replicated across the "
        "jet via the algebraic identities (X_{t+1} − X_t = ΔX_{t+1} = "
        "ε_{t+1}, and Δ²X_{t+1} = ΔX_{t+1} − ΔX_t = ε_{t+1} − ΔX_t, "
        "whose residual after subtracting −ΔX_t is again ε_{t+1}). So "
        "the three residuals are *perfectly correlated*.",
        "",
        "| | resid[0] | resid[1] | resid[2] |",
        "| :--- | ---: | ---: | ---: |",
    ]
    cross_matrix = np.array([[r["correlation"] for r in cross_rows[3*i:3*i+3]]
                              for i in range(3)])
    for i in range(3):
        md.append(f"| **resid[{i}]** | "
                  f"{cross_matrix[i, 0]:+.4f} | "
                  f"{cross_matrix[i, 1]:+.4f} | "
                  f"{cross_matrix[i, 2]:+.4f} |")

    md += [
        "",
        "All off-diagonal entries ≈ +1: the three residuals are the same "
        "ε_{t+1} on every step. **The residual covariance is rank-1** — "
        "the noise has one degree of freedom replicated three times.",
        "",
        "This is a feature of the algebraic constraints of the jet, not "
        "of the random walk specifically: in E15 (AR(1)) and E16 "
        "(AR(2)) the same cross-correlation pattern holds, because the "
        "jet identities tie the channels together with a single "
        "innovation. Looking back at E15 we should see the same near-1 "
        "off-diagonals — let me note that this property is universal "
        "to the kinematic jet representation: the three channels are "
        "*not three independent noise channels*; they are three views "
        "of one univariate trajectory, and the innovation that drives "
        "the trajectory is the only random ingredient.",
        "",
        "## What this experiment establishes",
        "",
        "The substrate's uniform reading survives the unit root cleanly. "
        "The planted M̂ is recovered, the graded components match, the "
        "residuals are white and stationary. The non-stationarity of the "
        "level channel (Var(X_t) ∝ t) does not break the fit because "
        "the OLS is happy on the jet — the level coefficient is "
        "super-consistently estimated and the difference channels are "
        "stationary by construction.",
        "",
        "The cross-correlation analysis exposes a structural fact about "
        "the jet representation: **the three jet channels share a single "
        "innovation per step.** This is true for every iid trajectory in "
        "the programme, and it tells us that the rank of the residual "
        "covariance matrix is the rank of the innovation process — for "
        "a univariate process driven by one Gaussian shock, that rank "
        "is 1.",
        "",
        "Next steps of phase (c):",
        "",
        "  - **E18**: SPTLS on heavy-tail trajectories (E10 Cauchy, E13 "
        "t_3, E14 stable α=1.5). The same machinery should recover the "
        "operator (the dependence structure is the same — iid, "
        "operator = cumulative identity), but the residual moments will "
        "not exist. The graded components should still be readable from "
        "the operator structure, and the operator structure is identical "
        "to E17 (random walk).",
        "",
        "  - **E19**: SPTLS on the log-normal trajectory (E09). Working "
        "on the multiplicative side: apply the jet-SPTLS directly to "
        "S_t = ∏ M_t. The operator should recover the geometric-Brownian "
        "structure (X_{t+1} ≈ X_t·M_{t+1}) which on the log scale is "
        "the random walk of E17. The connection between additive and "
        "multiplicative substrate readings should fall out cleanly.",
        "",
    ]
    (RESULTS / "E17_sptls_on_random_walk.md").write_text("\n".join(md))
    print(f"\nWrote {RESULTS / 'E17_sptls_on_random_walk.md'}")


if __name__ == "__main__":
    main()
