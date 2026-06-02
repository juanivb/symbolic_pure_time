# E12 — The AR(2) Gaussian trajectory  (φ₁ = 0.5, φ₂ = 0.3)

**Rule:** X_t = φ₁ X_{t-1} + φ₂ X_{t-2} + ε_t, with ε_t ~ N(0, 1) iid and (φ₁, φ₂) = (0.5, 0.3).  **Trajectory length:** N = 200000.

## What is new in this experiment

The dependence is now depth-2: the next state is a linear combination of the previous *two* states plus an independent Gaussian shock. A depth-1 operator (the AR(1) fit from E11) cannot represent this — there is a piece of the dependence (the lag-2 term) that has no slot in a depth-1 model. The central question of this experiment is: **how does the substrate tell you that depth-1 is the wrong depth?** The answer is concrete: look at the residual autocorrelation function of the fit. If it has a non-zero structure, the depth is wrong.

## The autocorrelation function follows the depth-2 recurrence

For a stationary AR(2), the autocorrelations at successive lags satisfy ρ_k = φ₁ ρ_{k−1} + φ₂ ρ_{k−2} for k ≥ 2, with the boundary value ρ_1 = φ₁/(1 − φ₂). Empirically:

| lag k | empirical ρ_k | predicted (recurrence) |
| ---: | ---: | ---: |
| 0 | +1.0000 | +1.0000 |
| 1 | +0.7121 | +0.7143 |
| 2 | +0.6564 | +0.6571 |
| 3 | +0.5416 | +0.5429 |
| 4 | +0.4673 | +0.4686 |
| 5 | +0.3973 | +0.3971 |
| 6 | +0.3409 | +0.3391 |

Match within sampling noise at every lag. The depth-2 recurrence is the right description of the dependence in the trajectory.

## Depth-1 fit: misses the depth-2 structure

Fitting the AR(1) model from E11 (OLS of X_t on X_{t-1}):

| quantity | empirical | implied / planted |
| :--- | ---: | ---: |
| φ̂                          | +0.7121 | ≈ ρ_1 = +0.7143 (NOT the planted φ₁ = +0.5000) |
| residual σ                  | 1.0507 | — (will be larger than the true σ = 1.0000 because the fit is mis-specified) |

**Critical reading:** the depth-1 fit gives `φ̂ = ρ_1`, the lag-1 autocorrelation. This is not equal to the planted φ₁ = 0.5. The depth-1 model interprets the entire dependence as a depth-1 effect and reads the lag-1 correlation as its parameter — but that correlation is shaped by both φ₁ and φ₂, so the reading is biased.

The decisive diagnostic is the residual ACF after the depth-1 fit. If the model captured all the dependence, the residuals would be white. Empirical residual ACF at lags 1 – 5:

| lag | residual ACF (depth-1 fit) |
| ---: | ---: |
| 1 | -0.2158 |
| 2 | +0.1959 |
| 3 | +0.0326 |
| 4 | +0.0723 |
| 5 | +0.0472 |

**The lag-1 residual ACF is *not* zero** — there is leftover depth-1 correlation that the depth-1 fit could not absorb. This is the certificate that depth-1 is the wrong depth: the fit looks plausible in its coefficient, but the residuals expose that it has missed something.

## Depth-2 fit: recovers the operator and leaves white residuals

Fitting OLS of X_t on (X_{t-1}, X_{t-2}):

| quantity | empirical | planted |
| :--- | ---: | ---: |
| φ̂₁                         | +0.4963 | +0.5000 |
| φ̂₂                         | +0.3030 | +0.3000 |
| residual σ                  | 1.0013 | 1.0000 |

Both planted coefficients are recovered to within sampling noise. The residual standard deviation matches the planted innovation standard deviation. Residual ACF at lags 1 – 5:

| lag | residual ACF (depth-2 fit) |
| ---: | ---: |
| 1 | -0.0000 |
| 2 | +0.0003 |
| 3 | -0.0003 |
| 4 | -0.0039 |
| 5 | +0.0001 |

All near zero. The depth-2 fit captures all the dependence; nothing is left for a depth-3 reading to find.

## Side-by-side residual ACFs (the diagnostic)

| lag | residual ACF (depth-1) | residual ACF (depth-2) |
| ---: | ---: | ---: |
| 1 | -0.2158 | -0.0000 |
| 2 | +0.1959 | +0.0003 |
| 3 | +0.0326 | -0.0003 |
| 4 | +0.0723 | -0.0039 |
| 5 | +0.0472 | +0.0001 |

Reading the rows: the depth-1 fit leaves a clear residual autocorrelation; the depth-2 fit leaves nothing measurable. **The residual ACF tells you whether the operator is at the right depth.** And from this diagnostic the right depth can be discovered iteratively: fit at depth k, check residual ACF, increase k if non-zero structure remains.

## The substrate's view of this trajectory

The operator is now a depth-2 object: two numbers (φ₁, φ₂) rather than one. Reading the operator at the right depth gives a faithful one-step prediction (the residuals are the pure innovation) and a complete characterisation of the process: from (φ₁, φ₂, σ) we can compute the stationary variance, all autocorrelations, and the variance of the running sum.

Reading at the wrong depth (depth-1) leaves a measurable trace in the residuals. That trace is itself information — it tells the substrate that more depth is needed. The procedure that follows is iterative: increase the depth until the residual ACF is white. This is the operational form of the depth-of-memory question, answerable directly from the trajectory.

## What this experiment shows

Together with E11 this establishes the depth-of-memory tools for the continuous-step case: when the operator is at the right depth, parameters are recovered cleanly and residuals are white; when it is at the wrong depth, the residual ACF makes the mismatch visible. The depth itself is therefore a measurable property of the trajectory, not a modelling choice that has to be made in advance.

The next experiment (E13) extends this same logic to a qualitatively different type of dependence — long-range memory, where the influence of past steps decays not geometrically (as in AR(p)) but at a slower rate. The depth-of-memory tools from E11 / E12 should still flag the dependence, but the residuals of any finite-depth AR fit will show a residual structure that no fixed depth can absorb. That is the qualitative new regime to investigate.
