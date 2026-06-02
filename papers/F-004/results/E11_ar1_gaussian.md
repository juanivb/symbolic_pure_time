# E11 — The AR(1) Gaussian trajectory  (φ = 0.7)

**Rule:** X_t = φ X_{t-1} + ε_t, with ε_t ~ N(0, 1) iid and φ = 0.7.  **Trajectory length:** N = 200000.  **Stationary marginal:** N(0, σ²/(1 − φ²)).

## What is new in this experiment

First trajectory with continuous-step memory. The Gaussian step of E06 has been carried over unchanged — the innovation ε_t is the same N(0, 1) draw — and a single new ingredient has been added: the term φ·X_{t-1} that couples X_t to its immediate past. This is the continuous-step analog of the sticky coin (E05), with the same depth of memory but a continuous state instead of a two-valued one.

## The stationary marginal

Empirical first two moments of the trajectory:

| quantity | empirical | expected |
| :--- | ---: | ---: |
| mean             | +0.0002 | 0 |
| variance         | 1.9559 | 1.9608 = σ²/(1−φ²) |
| L¹ (hist vs density) | 0.0116 | sampling noise |

The stationary marginal is exactly the Gaussian of E06, but with variance broadened by the memory feedback to σ²/(1 − φ²) ≈ 1.96 instead of 1. Same shape, wider.

## The autocorrelation function decays geometrically

Empirical ρ_k at lags 0..10 vs the prediction ρ_k = φ^k:

| lag k | empirical ρ_k | predicted φ^k |
| ---: | ---: | ---: |
| 0 | +1.0000 | +1.0000 |
| 1 | +0.6982 | +0.7000 |
| 2 | +0.4888 | +0.4900 |
| 3 | +0.3410 | +0.3430 |
| 4 | +0.2377 | +0.2401 |
| 5 | +0.1677 | +0.1681 |
| 6 | +0.1194 | +0.1176 |
| 7 | +0.0839 | +0.0824 |
| 8 | +0.0584 | +0.0576 |
| 9 | +0.0401 | +0.0404 |
| 10 | +0.0296 | +0.0282 |

Match within sampling noise at every lag. The correlation between X_t and X_{t+k} dies as φ^k = 0.7^k — at lag 10 it is already below 3%. The memory is short-range in absolute terms (a few steps suffice to forget) but it is still genuinely depth-1: at each step the next state is conditioned on the *previous* state, with all earlier states contributing only through that one.

## The variance of the running sum scales by a memory factor

If the steps were independent (no memory), the variance of S_n would simply be n · σ²_X = n · 1.96. With memory, the long-run variance per step is

    σ²_X · (1 + φ) / (1 − φ)  =  1.96 · 5.67  ≈  11.11,

a factor of 5.67 larger than the iid value. Empirically:

| horizon n | Var(S_n)/n (empirical) | iid prediction | memory-corrected prediction (long-run) |
| ---: | ---: | ---: | ---: |
| 1 | 1.9559 | 1.9608 | 11.1111 |
| 5 | 6.0262 | 1.9608 | 11.1111 |
| 20 | 9.6363 | 1.9608 | 11.1111 |
| 100 | 11.2322 | 1.9608 | 11.1111 |
| 1000 | 11.3039 | 1.9608 | 11.1111 |

Var(S_n)/n approaches the memory-corrected prediction at large n. At horizon 1 it equals σ²_X exactly (one step, no accumulation of memory). At horizons 100 – 1000 it sits at the long-run value. The persistence inflates the variance by ≈ 5.7× compared to the iid expectation. Same statement as in E05 — same machinery, different state space.

## The substrate's depth-1 operator is recovered cleanly

Fitting the depth-1 operator by OLS regression of X_t on X_{t-1}:

| parameter | empirical | planted |
| :--- | ---: | ---: |
| φ                                       | +0.6982 | +0.7000 |
| innovation σ (residual standard dev)   | 1.0012 | 1.0000 |

Residual autocorrelation function (after subtracting the depth-1 fit) at lags 1 – 5:

| lag | residual ACF |
| ---: | ---: |
| 1 | -0.0018 |
| 2 | +0.0029 |
| 3 | +0.0002 |
| 4 | -0.0032 |
| 5 | +0.0002 |

All residual autocorrelations are at sampling-noise level (~ 0.01). The depth-1 operator captures all the dependence in the trajectory; nothing is left for a depth-2 or deeper reading to find. **This is the certificate that the operator at the right depth captures the process fully** — when the trajectory is AR(1), the depth-1 fit's residuals are white.

## What this experiment shows

The dependence-axis tools we built for the discrete sticky coin in E05 transfer to continuous-state trajectories without change. The autocorrelation function decays geometrically as φ^k; the variance of the running sum is inflated by exactly the same (1+ρ)/(1−ρ) factor with ρ = φ; the depth-1 operator (a single number φ) is recovered by OLS and the residuals are white. The substrate's depth-1 reading is, for this process, the entire story.

The natural next question is what happens when the dependence is deeper than depth 1 — when the next step depends on more than just the immediately previous one. That is E12 (AR(2)), where the depth-1 reading is no longer enough and the residuals expose the missing structure.
