# E15 — SPTLS on the kinematic jet of the AR(1) trajectory  (φ = 0.7)

**First experiment of phase (c): connect the empirical programme to the substrate language.**  We take the AR(1) trajectory of E11 (φ = 0.7, ε ~ N(0, 1) iid) and apply the substrate's natural reading — the SPTLS one-step operator on the kinematic jet q_t = (X_t, ΔX_t, Δ²X_t). The point is to verify that this uniform machinery reads, on this clean stationary case, exactly the same content we already extracted in E11 by the simpler depth-1 fit.

## Why this experiment matters for the programme

Every previous experiment used the natural reading for its case: counts of coin flips, transition matrices for Markov, lag-coefficient OLS for AR(p). The substrate's reading is uniform across all cases: take the jet and fit one 3 × 3 operator. This experiment is the first verification that the uniform reading matches the natural one when both are clear. Once that match is in place, the substrate reading can be applied to cases where no obvious natural reading exists — general multivariate processes, processes without preferred lags, processes with hybrid dependence structure.

## The planted operator and its recovery

For an AR(1) trajectory the algebraic identities of the jet (ΔX_{t+1} = X_{t+1} − X_t and Δ²X_{t+1} = ΔX_{t+1} − ΔX_t) force the SPTLS operator on the jet to have a specific form. Substituting X_{t+1} = φ X_t + ε_{t+1}:

    M̂_planted  =  [[ φ,       0,    0],  =  [[ +0.7000,  0.0,   0.0],
                   [ φ − 1,   0,    0],     [ -0.3000, 0.0,   0.0],
                   [ φ − 1, −1,     0]]     [ -0.3000, -1.0,  0.0]]

Empirically, with N = 200 000 samples, the OLS fit on the jet recovers:

| row | col | M̂ empirical | M̂ planted | |diff| |
| :--- | :--- | ---: | ---: | ---: |
| 0 | 0 | +0.6983 | +0.7000 | 0.0017 |
| 0 | 1 | +0.0003 | +0.0000 | 0.0003 |
| 0 | 2 | -0.0022 | +0.0000 | 0.0022 |
| 1 | 0 | -0.3017 | -0.3000 | 0.0017 |
| 1 | 1 | +0.0003 | +0.0000 | 0.0003 |
| 1 | 2 | -0.0022 | +0.0000 | 0.0022 |
| 2 | 0 | -0.3017 | -0.3000 | 0.0017 |
| 2 | 1 | -0.9997 | -1.0000 | 0.0003 |
| 2 | 2 | -0.0022 | +0.0000 | 0.0022 |

Frobenius distance ‖M̂_emp − M̂_planted‖_F = 0.0049, at sampling-noise level. **The substrate machinery recovers the planted operator without ambiguity.**

## The four graded readings of M̂

The substrate decomposes the operator into four pieces, each with a geometric meaning. Empirical values vs the analytical values implied by the planted M̂ at φ = 0.7:

| component | empirical | planted |
| :--- | ---: | ---: |
| `s0_trace_over_3` | +0.2321 | +0.2333 |
| `rotor_energy_frobenius_squared` | +0.5879 | +0.5900 |
| `e12_bivector` | +0.1510 | +0.1500 |
| `e13_bivector` | +0.1497 | +0.1500 |
| `e23_bivector` | +0.4987 | +0.5000 |
| `spin2_energy_frobenius_squared` | +0.9194 | +0.9167 |
| `det_M` | -0.0022 | +0.0000 |

Match is at sampling-noise level on every component. The operator's structure for AR(1) in jet coordinates is non-trivial in all four grades: trace/3 = φ/3 reads the AR coefficient itself, the rotor part carries the cross-channel coupling implied by the algebraic constraints of the jet, the spin-2 part carries the symmetric traceless residual, and the determinant is zero (the operator is singular because the column for Δ²X_t has no predictive content for the next jet — the AR(1) only sees X_t).

## Residual diagnostics per channel

After fitting M̂, each channel of the residual q_{t+1} − M̂ q_t should be near-zero on average and white on autocorrelation. The variance of the residual on the level channel should equal the innovation variance σ² = 1.

| channel | residual mean | residual variance | ACF lags 1..5 |
| :--- | ---: | ---: | :--- |
| level (X_t) | +0.0001 | 1.0024 | [-0.0, 0.0, 0.0004, -0.0031, 0.0003] |
| Δ velocity (ΔX_t) | +0.0001 | 1.0024 | [-0.0, 0.0, 0.0004, -0.0031, 0.0003] |
| Δ² acceleration (Δ²X_t) | +0.0001 | 1.0024 | [-0.0, 0.0, 0.0004, -0.0031, 0.0003] |

All three channels have residuals with mean ≈ 0 and white ACF (all lag-k autocorrelations at sampling-noise level, |ACF| ≈ 0.003 or less). The level-channel residual has variance ≈ 1 = σ², the planted innovation variance. The other channels have residual variances tied to σ² by the algebraic identities (ΔX_{t+1} = X_{t+1} − X_t inherits the innovation directly; Δ²X_{t+1} = innovation minus ΔX_t, so its residual variance is the same as the level's). **The fit is clean: the substrate operator captures all the structure, the residuals are the innovation alone.**

## What this experiment establishes

The substrate's uniform reading (kinematic jet + one 3 × 3 operator + four graded components) reproduces, on the AR(1) stationary case, exactly the same content that E11 extracted via the simpler depth-1 OLS. The two readings agree numerically to within sampling noise. **The substrate machinery has been validated on a case where we knew the answer.** From here, the same machinery extends without modification to:

  - AR(p) for p > 1 (E12 will be the next case to verify);

  - non-stationary trajectories (E00 - E03 running sums, E06 Gaussian walk — where the level has a unit root and the jet reading has to handle the degeneracy);

  - heavy-tail trajectories (E10, E13, E14 — where moments of the residual fail and the substrate must read graded coefficients quantile-based);

  - multiplicative trajectories (E09 log-normal — where the substrate reads the additive operator on log X_t).

Each of these is a separate experiment of phase (c), each verifying that the substrate's uniform machinery handles the case correctly. E15 sets the template: build the jet, fit M̂, read the four graded components, verify residuals are white, compare to whatever case-specific reading we have from earlier experiments.
