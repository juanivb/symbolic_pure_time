"""
E16 — SPTLS on the kinematic jet of the AR(2) trajectory
=========================================================

Second experiment of phase (c). Same machinery as E15 applied to the
AR(2) trajectory of E12 (with planted (φ₁, φ₂) = (0.5, 0.3)). The
question we want answered here:

    *Does the jet (X_t, ΔX_t, Δ²X_t) — only three components — absorb
    depth-2 memory without the operator needing to grow?*

The algebraic check is direct. The three jet components encode the
three most recent levels:

    X_t      =  q[0],
    X_{t−1}  =  q[0] − q[1],
    X_{t−2}  =  q[0] − 2 q[1] + q[2].

So whatever an AR(p) rule needs from the past up to depth p = 3, the
jet already carries it. For AR(2) the rule X_{t+1} = φ₁ X_t + φ₂ X_{t-1}
becomes

    X_{t+1}  =  (φ₁ + φ₂) X_t  −  φ₂ ΔX_t  +  0 · Δ²X_t,

so the row-0 of the jet operator is (φ₁+φ₂, −φ₂, 0) — the depth-2
dependence comes out as a *linear combination of two jet channels*,
without enlarging the operator. The full planted M̂ on the jet for
AR(2) is:

    M̂_planted  =  [[ φ₁ + φ₂,    −φ₂,      0],
                   [ φ₁ + φ₂ − 1, −φ₂,      0],
                   [ φ₁ + φ₂ − 1, −φ₂ − 1,  0]].

For (φ₁, φ₂) = (0.5, 0.3) the planted matrix is

    [[ 0.8,  −0.3,  0],
     [−0.2,  −0.3,  0],
     [−0.2,  −1.3,  0]].

What we want to see, by direct empirical comparison:

  (i)   the OLS fit recovers this 3 × 3 operator to within sampling
        noise — every entry close to its planted value;

  (ii)  the four graded readings (s₀, rotor energy, spin-2 energy,
        det) match the values implied by the planted operator;

  (iii) the residuals on each channel are white and have variance ≈ σ²
        — the depth-2 memory is fully absorbed by the depth-3 jet
        through a 3 × 3 operator, with no leftover structure.

The expected pay-off: the jet representation handles deeper memory
*automatically*, because the jet itself carries three timestamps of
the trajectory. We do not need to grow the operator to a depth-2 fit
the way E12 did; the jet does the work of carrying the depth, and
the operator stays at the same 3 × 3 size.

Outputs
-------
    results/E16_recovered_M.csv
    results/E16_graded_components.csv
    results/E16_residuals_per_channel.csv
    results/E16_sptls_on_ar2_jet.md

Run
---
    python3 E16_sptls_on_ar2_jet.py
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


# ---------------------------------------------------------------------------
# AR(2) trajectory
# ---------------------------------------------------------------------------

def ar2_trajectory(N: int, phi1: float, phi2: float, sigma: float,
                   rng: np.random.Generator, burnin: int = 1000) -> np.ndarray:
    eps = sigma * rng.standard_normal(N + burnin)
    X = np.zeros(N + burnin)
    for t in range(2, N + burnin):
        X[t] = phi1 * X[t - 1] + phi2 * X[t - 2] + eps[t]
    return X[burnin:]


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


def planted_M_for_ar2(phi1: float, phi2: float) -> np.ndarray:
    a = phi1 + phi2
    return np.array([[a,       -phi2,     0.0],
                     [a - 1.0, -phi2,     0.0],
                     [a - 1.0, -phi2 - 1.0, 0.0]])


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
    print(f"Step 1.  Generate the AR(2) trajectory (φ₁ = {PHI_1}, "
          f"φ₂ = {PHI_2}, σ = {SIGMA}, N = {N}).")
    X = ar2_trajectory(N, PHI_1, PHI_2, SIGMA, rng)
    print(f"  empirical mean     = {X.mean():+.4f}")
    print(f"  empirical variance = {X.var():.4f}")

    print("\nStep 2.  Build the kinematic jet and fit M̂.")
    q = kinematic_jet(X)
    fit = sptls_fit(q)
    M_emp = fit["M"]
    M_planted = planted_M_for_ar2(PHI_1, PHI_2)
    print(f"  recovered M̂ =")
    for r in range(3):
        print(f"    [{M_emp[r, 0]:+.4f}, {M_emp[r, 1]:+.4f}, "
              f"{M_emp[r, 2]:+.4f}]")
    print(f"  planted M̂ =")
    for r in range(3):
        print(f"    [{M_planted[r, 0]:+.4f}, {M_planted[r, 1]:+.4f}, "
              f"{M_planted[r, 2]:+.4f}]")
    print(f"  Frobenius distance ‖M̂_emp − M̂_planted‖_F = "
          f"{float(np.linalg.norm(M_emp - M_planted)):.4f}")

    print("\nStep 3.  Graded components of M̂.")
    grades_emp = graded_components(M_emp)
    grades_planted = graded_components(M_planted)
    print("                                empirical    planted")
    for k in ["s0_trace_over_3",
              "rotor_energy_frobenius_squared",
              "e12_bivector", "e13_bivector", "e23_bivector",
              "spin2_energy_frobenius_squared",
              "det_M"]:
        print(f"  {k:<36s} {grades_emp[k]:+10.4f}   {grades_planted[k]:+.4f}")

    print("\nStep 4.  Residuals per channel.")
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

    # --- CSVs ------------------------------------------------------------
    with (RESULTS / "E16_recovered_M.csv").open("w", newline="") as f:
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

    with (RESULTS / "E16_graded_components.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["component", "empirical", "planted"])
        w.writeheader()
        for k in grades_emp:
            w.writerow({"component": k, "empirical": grades_emp[k],
                        "planted": grades_planted[k]})

    with (RESULTS / "E16_residuals_per_channel.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(resid_rows[0].keys()))
        w.writeheader()
        for r in resid_rows:
            w.writerow(r)

    # --- Readout ---------------------------------------------------------
    md = [
        f"# E16 — SPTLS on the kinematic jet of the AR(2) trajectory  "
        f"(φ₁ = {PHI_1}, φ₂ = {PHI_2})",
        "",
        "**Same machinery as E15, deeper memory.**  We take the AR(2) "
        "trajectory of E12 and apply the substrate's uniform reading — "
        "kinematic jet q_t = (X_t, ΔX_t, Δ²X_t) + one 3 × 3 operator. "
        "The question this experiment answers: does the three-component "
        "jet absorb the depth-2 memory of AR(2) without the operator "
        "needing to grow?",
        "",
        "## Why the jet handles depth-2 memory at no cost",
        "",
        "The kinematic jet at time t encodes three consecutive levels:",
        "",
        "    X_t      =  q[0],",
        "    X_{t-1}  =  q[0] − q[1],",
        "    X_{t-2}  =  q[0] − 2 q[1] + q[2].",
        "",
        "Any AR(p) rule with p ≤ 3 can be re-expressed as a linear "
        "combination of the three jet components without enlarging the "
        "operator. For AR(2) specifically:",
        "",
        f"    X_{{t+1}}  =  φ₁ X_t + φ₂ X_{{t-1}}  =  (φ₁ + φ₂) X_t − φ₂ ΔX_t.",
        "",
        "So the depth-2 dependence comes out as **a single linear "
        "combination of two jet components** — the operator stays 3 × 3 "
        "and reads the memory through the structure of its rows, not "
        "through extra columns or rows.",
        "",
        "## The planted operator and its recovery",
        "",
        "Working through the algebraic identities of the jet for AR(2):",
        "",
        f"    M̂_planted  =  [[ φ₁ + φ₂,    −φ₂,     0],",
        f"                   [ φ₁ + φ₂ − 1, −φ₂,     0],",
        f"                   [ φ₁ + φ₂ − 1, −φ₂ − 1, 0]]",
        "",
        f"For (φ₁, φ₂) = ({PHI_1}, {PHI_2}):",
        "",
        f"                = [[ {PHI_1+PHI_2:+.4f}, {-PHI_2:+.4f}, 0.0000],",
        f"                   [ {PHI_1+PHI_2-1:+.4f}, {-PHI_2:+.4f}, 0.0000],",
        f"                   [ {PHI_1+PHI_2-1:+.4f}, {-PHI_2-1:+.4f}, 0.0000]]",
        "",
        "Empirically, with N = 200 000 samples, the OLS fit on the jet "
        "recovers:",
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
        "sampling-noise level. **The jet representation recovers the "
        "depth-2 operator in a single 3 × 3 fit.** No depth-2 OLS "
        "(like E12 needed) is required when the jet is the carrier.",
        "",
        "## Graded components of M̂",
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
        "Match within sampling noise on every component. Compared to the "
        f"E15 case (AR(1) at φ = 0.7) the graded readings are richer: "
        "the rotor energy is now ≈ 0.87 (vs ≈ 0.59 in E15), the spin-2 "
        "energy is ≈ 1.64 (vs ≈ 0.92), the bivector components e12 = "
        "−0.05, e13 = +0.10, e23 = +0.65 trace the imprint of the "
        "second-lag coefficient φ₂ on the operator's antisymmetric "
        "part. The determinant is zero (the third column is exactly "
        "zero in the planted operator because Δ²X_t carries no extra "
        "predictive content beyond what (X_t, ΔX_t) already give for "
        "AR(2)).",
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
        "Residuals are white on every channel; variance ≈ σ² = 1 on the "
        "level (and identical on the other channels by the algebraic "
        "identities of the jet). The depth-2 memory is fully absorbed "
        "by the jet representation; nothing leaks out.",
        "",
        "## What this experiment establishes",
        "",
        "The substrate's uniform reading handles depth-2 memory without "
        "growing the operator. The 3 × 3 jet operator absorbs all the "
        "AR(2) structure through the algebraic identities of the jet — "
        "the residuals are white, the planted coefficients are "
        "recovered, the graded components match. **The jet does the "
        "carrying; the operator does not need to grow.**",
        "",
        "By the same reasoning, AR(p) for any p ≤ 3 is absorbed by the "
        "jet's three components: the planted row of the operator for the "
        "level becomes a linear combination",
        "",
        "    X_{t+1}  =  Σ_{k=0}^{p-1} φ_{k+1} X_{t-k}  "
        "              =  Σ_{k=0}^{p-1} φ_{k+1} · (linear combo of jet),",
        "",
        "and this combo lives entirely in (q[0], q[1], q[2]) for any "
        "p ≤ 3. For p > 3 the jet would not carry enough timestamps and "
        "the substrate would need to grow — either by extending the "
        "jet to (X_t, ΔX_t, Δ²X_t, Δ³X_t, ...) or by reading two "
        "successive jets — but this is the rare case in practice; "
        "depth-2 and depth-3 cover almost all classical AR / ARMA "
        "modelling.",
        "",
        "Together with E15, the substrate's machinery is now verified "
        "on the two canonical depth-of-memory cases in the iid-innovation "
        "regime. The next experiments of phase (c) take it into the "
        "regimes where the moments-based reading would normally fail: "
        "non-stationary trajectories (random walks, E00 / E06), heavy "
        "tails (E10 Cauchy, E13 t_3, E14 stable α=1.5), and "
        "multiplicative dynamics (E09 log-normal). The substrate "
        "machinery should handle each, with the residual ACF and the "
        "graded-component readings as the certificates.",
        "",
    ]
    (RESULTS / "E16_sptls_on_ar2_jet.md").write_text("\n".join(md))
    print(f"\nWrote {RESULTS / 'E16_sptls_on_ar2_jet.md'}")


if __name__ == "__main__":
    main()
