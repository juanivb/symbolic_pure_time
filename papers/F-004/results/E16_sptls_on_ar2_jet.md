# E16 — SPTLS on the kinematic jet of the AR(2) trajectory  (φ₁ = 0.5, φ₂ = 0.3)

**Same machinery as E15, deeper memory.**  We take the AR(2) trajectory of E12 and apply the substrate's uniform reading — kinematic jet q_t = (X_t, ΔX_t, Δ²X_t) + one 3 × 3 operator. The question this experiment answers: does the three-component jet absorb the depth-2 memory of AR(2) without the operator needing to grow?

## Why the jet handles depth-2 memory at no cost

The kinematic jet at time t encodes three consecutive levels:

    X_t      =  q[0],
    X_{t-1}  =  q[0] − q[1],
    X_{t-2}  =  q[0] − 2 q[1] + q[2].

Any AR(p) rule with p ≤ 3 can be re-expressed as a linear combination of the three jet components without enlarging the operator. For AR(2) specifically:

    X_{t+1}  =  φ₁ X_t + φ₂ X_{t-1}  =  (φ₁ + φ₂) X_t − φ₂ ΔX_t.

So the depth-2 dependence comes out as **a single linear combination of two jet components** — the operator stays 3 × 3 and reads the memory through the structure of its rows, not through extra columns or rows.

## The planted operator and its recovery

Working through the algebraic identities of the jet for AR(2):

    M̂_planted  =  [[ φ₁ + φ₂,    −φ₂,     0],
                   [ φ₁ + φ₂ − 1, −φ₂,     0],
                   [ φ₁ + φ₂ − 1, −φ₂ − 1, 0]]

For (φ₁, φ₂) = (0.5, 0.3):

                = [[ +0.8000, -0.3000, 0.0000],
                   [ -0.2000, -0.3000, 0.0000],
                   [ -0.2000, -1.3000, 0.0000]]

Empirically, with N = 200 000 samples, the OLS fit on the jet recovers:

| row | col | M̂ empirical | M̂ planted | |diff| |
| :--- | :--- | ---: | ---: | ---: |
| 0 | 0 | +0.7993 | +0.8000 | 0.0007 |
| 0 | 1 | -0.3031 | -0.3000 | 0.0031 |
| 0 | 2 | +0.0000 | +0.0000 | 0.0000 |
| 1 | 0 | -0.2007 | -0.2000 | 0.0007 |
| 1 | 1 | -0.3031 | -0.3000 | 0.0031 |
| 1 | 2 | +0.0000 | +0.0000 | 0.0000 |
| 2 | 0 | -0.2007 | -0.2000 | 0.0007 |
| 2 | 1 | -1.3031 | -1.3000 | 0.0031 |
| 2 | 2 | +0.0000 | +0.0000 | 0.0000 |

Frobenius distance ‖M̂_emp − M̂_planted‖_F = 0.0055, at sampling-noise level. **The jet representation recovers the depth-2 operator in a single 3 × 3 fit.** No depth-2 OLS (like E12 needed) is required when the jet is the carrier.

## Graded components of M̂

| component | empirical | planted |
| :--- | ---: | ---: |
| `s0_trace_over_3` | +0.1654 | +0.1667 |
| `rotor_energy_frobenius_squared` | +0.8745 | +0.8700 |
| `e12_bivector` | -0.0512 | -0.0500 |
| `e13_bivector` | +0.1004 | +0.1000 |
| `e23_bivector` | +0.6516 | +0.6500 |
| `spin2_energy_frobenius_squared` | +1.6447 | +1.6367 |
| `det_M` | +0.0000 | +0.0000 |

Match within sampling noise on every component. Compared to the E15 case (AR(1) at φ = 0.7) the graded readings are richer: the rotor energy is now ≈ 0.87 (vs ≈ 0.59 in E15), the spin-2 energy is ≈ 1.64 (vs ≈ 0.92), the bivector components e12 = −0.05, e13 = +0.10, e23 = +0.65 trace the imprint of the second-lag coefficient φ₂ on the operator's antisymmetric part. The determinant is zero (the third column is exactly zero in the planted operator because Δ²X_t carries no extra predictive content beyond what (X_t, ΔX_t) already give for AR(2)).

## Residuals per channel

| channel | residual mean | residual variance | ACF lags 1..5 |
| :--- | ---: | ---: | :--- |
| level (X_t) | +0.0003 | 1.0027 | [-0.0, 0.0003, -0.0004, -0.0039, 0.0] |
| Δ velocity (ΔX_t) | +0.0003 | 1.0027 | [-0.0, 0.0003, -0.0004, -0.0039, 0.0] |
| Δ² acceleration (Δ²X_t) | +0.0003 | 1.0027 | [-0.0, 0.0003, -0.0004, -0.0039, 0.0] |

Residuals are white on every channel; variance ≈ σ² = 1 on the level (and identical on the other channels by the algebraic identities of the jet). The depth-2 memory is fully absorbed by the jet representation; nothing leaks out.

## What this experiment establishes

The substrate's uniform reading handles depth-2 memory without growing the operator. The 3 × 3 jet operator absorbs all the AR(2) structure through the algebraic identities of the jet — the residuals are white, the planted coefficients are recovered, the graded components match. **The jet does the carrying; the operator does not need to grow.**

By the same reasoning, AR(p) for any p ≤ 3 is absorbed by the jet's three components: the planted row of the operator for the level becomes a linear combination

    X_{t+1}  =  Σ_{k=0}^{p-1} φ_{k+1} X_{t-k}                =  Σ_{k=0}^{p-1} φ_{k+1} · (linear combo of jet),

and this combo lives entirely in (q[0], q[1], q[2]) for any p ≤ 3. For p > 3 the jet would not carry enough timestamps and the substrate would need to grow — either by extending the jet to (X_t, ΔX_t, Δ²X_t, Δ³X_t, ...) or by reading two successive jets — but this is the rare case in practice; depth-2 and depth-3 cover almost all classical AR / ARMA modelling.

Together with E15, the substrate's machinery is now verified on the two canonical depth-of-memory cases in the iid-innovation regime. The next experiments of phase (c) take it into the regimes where the moments-based reading would normally fail: non-stationary trajectories (random walks, E00 / E06), heavy tails (E10 Cauchy, E13 t_3, E14 stable α=1.5), and multiplicative dynamics (E09 log-normal). The substrate machinery should handle each, with the residual ACF and the graded-component readings as the certificates.
