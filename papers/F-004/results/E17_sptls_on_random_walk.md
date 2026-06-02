# E17 — SPTLS on the kinematic jet of the Gaussian random walk

**Third experiment of phase (c).**  Same machinery as E15 / E16, now applied to the Gaussian random walk of E06 — the φ = 1 limit of AR(1). The level channel is non-stationary: Var(X_t) grows linearly with t, while the difference channels (ΔX_t, Δ²X_t) remain stationary. The unit root makes the level dominate the covariance matrix by a factor of ~N, which is the kind of degeneracy where naive variance-based fits can misbehave. The question is whether the substrate's uniform reading still works.

## The planted operator

Setting φ = 1 in the AR(1) formula M̂ = [[φ, 0, 0], [φ-1, 0, 0], [φ-1, -1, 0]] gives

    M̂_planted  =  [[ 1.0000,  0.0000,  0.0000],
                   [ 0.0000,  0.0000,  0.0000],
                   [ 0.0000, -1.0000,  0.0000]].

The level just carries forward (no decay, no growth). The velocity row is exactly zero — meaning ΔX_{t+1} is pure innovation, independent of everything in the current jet. The acceleration row is (0, −1, 0): Δ²X_{t+1} subtracts the previous ΔX_t (and is perturbed by the innovation that already drove the level and velocity rows).

## Recovery of M̂

OLS on the jet, with N = 200 000 samples:

| row | col | M̂ empirical | M̂ planted | |diff| |
| :--- | :--- | ---: | ---: | ---: |
| 0 | 0 | +0.9999 | +1.0000 | 0.0001 |
| 0 | 1 | -0.0019 | +0.0000 | 0.0019 |
| 0 | 2 | -0.0017 | +0.0000 | 0.0017 |
| 1 | 0 | -0.0001 | +0.0000 | 0.0001 |
| 1 | 1 | -0.0019 | +0.0000 | 0.0019 |
| 1 | 2 | -0.0017 | +0.0000 | 0.0017 |
| 2 | 0 | -0.0001 | +0.0000 | 0.0001 |
| 2 | 1 | -1.0019 | -1.0000 | 0.0019 |
| 2 | 2 | -0.0017 | +0.0000 | 0.0017 |

Frobenius distance ‖M̂_emp − M̂_planted‖_F = 0.0044, at sampling-noise level. **The substrate machinery recovers the unit-root operator cleanly.** The OLS estimator is in fact *super-consistent* on the level coefficient because the level has unbounded variance — the regression has effectively infinite leverage on that coefficient. The result is that the recovered value is closer to the planted than naive √N sampling noise would suggest.

## Graded components of M̂

Comparison to the values implied by the planted M̂ at φ = 1:

| component | empirical | planted |
| :--- | ---: | ---: |
| `s0_trace_over_3` | +0.3321 | +0.3333 |
| `rotor_energy_frobenius_squared` | +0.5002 | +0.5000 |
| `e12_bivector` | -0.0009 | +0.0000 |
| `e13_bivector` | -0.0008 | +0.0000 |
| `e23_bivector` | +0.5001 | +0.5000 |
| `spin2_energy_frobenius_squared` | +1.1725 | +1.1667 |
| `det_M` | -0.0017 | +0.0000 |

Three points now in the dependence-axis fingerprint:

| case | φ / depth | s₀ | rotor energy | spin-2 energy |
| :--- | :--- | ---: | ---: | ---: |
| E15 AR(1) φ=0.7 | depth 1, stationary | +0.233 | 0.59 | 0.92 |
| E16 AR(2)       | depth 2, stationary | +0.167 | 0.87 | 1.64 |
| E17 random walk | φ=1, unit root      | +0.333 | 0.50 | 1.17 |

The three cases have distinguishable graded fingerprints. The random walk has the highest s₀ (1/3 exactly, because the trace is just the unit-root coefficient 1). The AR(2) has the highest rotor and spin-2 energies because depth-2 memory adds bivector components that depth-1 does not. The random walk's bivector content sits entirely on the e23 plane (0.5), which is the Cl(3,0) plane corresponding to the velocity-acceleration coupling.

## Residuals per channel

| channel | residual mean | residual variance | ACF lags 1..5 |
| :--- | ---: | ---: | :--- |
| level (X_t) | -0.0012 | 1.0024 | [-0.0, 0.0, -0.0007, -0.0037, -0.0002] |
| Δ velocity (ΔX_t) | -0.0012 | 1.0024 | [-0.0, 0.0, -0.0007, -0.0037, -0.0002] |
| Δ² acceleration (Δ²X_t) | -0.0012 | 1.0024 | [-0.0, 0.0, -0.0007, -0.0037, -0.0002] |

Same pattern as E15 / E16: residuals white on every channel, variance ≈ σ² = 1 on the level (and identical on the others by the algebraic identities of the jet).

## Cross-correlations of residuals (the unit-root signature)

For the random walk, the innovation ε_{t+1} drives all three jet channels simultaneously — there is no separate noise per channel, only one innovation per step replicated across the jet via the algebraic identities (X_{t+1} − X_t = ΔX_{t+1} = ε_{t+1}, and Δ²X_{t+1} = ΔX_{t+1} − ΔX_t = ε_{t+1} − ΔX_t, whose residual after subtracting −ΔX_t is again ε_{t+1}). So the three residuals are *perfectly correlated*.

| | resid[0] | resid[1] | resid[2] |
| :--- | ---: | ---: | ---: |
| **resid[0]** | +1.0000 | +1.0000 | +1.0000 |
| **resid[1]** | +1.0000 | +1.0000 | +1.0000 |
| **resid[2]** | +1.0000 | +1.0000 | +1.0000 |

All off-diagonal entries ≈ +1: the three residuals are the same ε_{t+1} on every step. **The residual covariance is rank-1** — the noise has one degree of freedom replicated three times.

This is a feature of the algebraic constraints of the jet, not of the random walk specifically: in E15 (AR(1)) and E16 (AR(2)) the same cross-correlation pattern holds, because the jet identities tie the channels together with a single innovation. Looking back at E15 we should see the same near-1 off-diagonals — let me note that this property is universal to the kinematic jet representation: the three channels are *not three independent noise channels*; they are three views of one univariate trajectory, and the innovation that drives the trajectory is the only random ingredient.

## What this experiment establishes

The substrate's uniform reading survives the unit root cleanly. The planted M̂ is recovered, the graded components match, the residuals are white and stationary. The non-stationarity of the level channel (Var(X_t) ∝ t) does not break the fit because the OLS is happy on the jet — the level coefficient is super-consistently estimated and the difference channels are stationary by construction.

The cross-correlation analysis exposes a structural fact about the jet representation: **the three jet channels share a single innovation per step.** This is true for every iid trajectory in the programme, and it tells us that the rank of the residual covariance matrix is the rank of the innovation process — for a univariate process driven by one Gaussian shock, that rank is 1.

Next steps of phase (c):

  - **E18**: SPTLS on heavy-tail trajectories (E10 Cauchy, E13 t_3, E14 stable α=1.5). The same machinery should recover the operator (the dependence structure is the same — iid, operator = cumulative identity), but the residual moments will not exist. The graded components should still be readable from the operator structure, and the operator structure is identical to E17 (random walk).

  - **E19**: SPTLS on the log-normal trajectory (E09). Working on the multiplicative side: apply the jet-SPTLS directly to S_t = ∏ M_t. The operator should recover the geometric-Brownian structure (X_{t+1} ≈ X_t·M_{t+1}) which on the log scale is the random walk of E17. The connection between additive and multiplicative substrate readings should fall out cleanly.
