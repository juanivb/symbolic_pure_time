# E19 — SPTLS on the log-normal trajectory: two coordinates, two stories

**Trajectory:** S_t = exp(W_t) where W_t = X_1 + X_2 + ... + X_t with X_i ~ N(0, σ²) iid, σ = 0.05.  **Length:** N = 100000.  **Two readings on the same trajectory:**  the additive (log) side W_t = log S_t, and the multiplicative side S_t.

## What is new in this experiment

Every previous (c) experiment fed the substrate machinery a trajectory in the coordinate where the underlying operation is additive. Here we have a trajectory whose natural operation is *multiplicative* — successive states are products, not sums. On the original scale (S_t) the operation does not linearise; on the log scale (W_t = log S_t) it does. The substrate's machinery only reads the linear / additive structure cleanly. We run it on both sides to see what happens.

## Additive (log) side — same as E17, exactly

Frobenius distance ‖M̂_log − M̂_planted‖_F = **0.0095**, matching the precision of E17. The recovered M̂ on the log side:

| | col 0 | col 1 | col 2 |
| :--- | ---: | ---: | ---: |
| row 0 | +0.9999 | +0.0023 | -0.0050 |
| row 1 | -0.0001 | +0.0023 | -0.0050 |
| row 2 | -0.0001 | -0.9977 | -0.0050 |

Graded readings: s₀ = +0.3324, rotor energy = 0.4928, spin-2 energy = 1.1711, det = -0.004963. **Identical to E17 within sampling noise.**

Residuals on the level channel: mean ≈ +0.000014, variance = 0.002501, **expected σ² = 0.002500**. ACF at lags 1..5: [1e-05, 2e-05, -0.002, -0.00439, -0.00415] — white at sampling-noise level.

On the additive side, the substrate machinery sees a Gaussian random walk with step variance σ² = 0.002500. Nothing distinguishes this experiment from E17 in the additive coordinate — only the scale of σ. The log-normal trajectory *is* a Gaussian walk in this view.

## Multiplicative side — the same operator structure, but distorted

Now applying the same machinery directly to S_t (without taking the log):

| | col 0 | col 1 | col 2 |
| :--- | ---: | ---: | ---: |
| row 0 | +0.9988 | -0.0619 | +0.0271 |
| row 1 | -0.0012 | -0.0619 | +0.0271 |
| row 2 | -0.0012 | -1.0619 | +0.0271 |

Frobenius distance to the planted random-walk M̂ = **0.1170**. Graded readings: s₀ = +0.3213, rotor energy = 0.5952, spin-2 energy = 1.2301, det = +0.027096.

Because the per-step log-return σ is small, S_{t+1} = S_t · exp(X_{t+1}) ≈ S_t · (1 + X_{t+1}) and the multiplicative trajectory looks *locally* like an additive random walk with level-dependent step. To first order in σ the level-row of M̂ on the multiplicative side is still (1, 0, 0): each step carries the level forward.

**But the residuals are not the same**. Here is where the wrong-coordinate cost shows up.

## Heteroskedasticity: the residuals on the multiplicative side scale with the level

Binning the trajectory by |S_t| (deciles) and reading the residual standard deviation in each bin:

| bin | |S_t| range | n | residual std | residual std / median |S_t| |
| ---: | :--- | ---: | ---: | ---: |
| 0 | [0.0001, 0.0110] | 10000 | 0.00021 | 0.21209 |
| 1 | [0.0110, 0.0713] | 10000 | 0.00216 | 0.05660 |
| 2 | [0.0713, 0.2247] | 9999 | 0.00693 | 0.05761 |
| 3 | [0.2247, 0.6832] | 10000 | 0.02223 | 0.05391 |
| 4 | [0.6832, 2.2906] | 9999 | 0.06460 | 0.05683 |
| 5 | [2.2906, 12.7457] | 10000 | 0.31999 | 0.06075 |
| 6 | [12.7457, 52.3156] | 10000 | 1.61765 | 0.05715 |
| 7 | [52.3156, 153.6284] | 9999 | 4.77648 | 0.05627 |
| 8 | [153.6284, 604.3820] | 10000 | 18.69992 | 0.05767 |
| 9 | [604.3820, 14975.6783] | 10000 | 141.62287 | 0.11486 |

The residual standard deviation grows with the bin's typical |S_t|, while the **ratio residual_std / |S_t|** stays roughly constant near σ = 0.05. **This is the empirical statement of "the noise on the multiplicative side is proportional to the level"**, i.e. multiplicative noise. The OLS-on-jet interpretation is therefore mis-specified at the residual level: it returns a usable level-row coefficient, but the moment-based summary of the residual (a single σ) is wrong because there is no single σ — only σ(S_t) = S_t · σ.

## Heteroskedasticity check on the additive (log) side  (should be flat)

Same binning, but on the log trajectory:

| bin | |W_t| range | n | residual std |
| ---: | :--- | ---: | ---: |
| 0 | [0.0000, 0.5358] | 10000 | 0.04969 |
| 1 | [0.5358, 1.1898] | 10000 | 0.05008 |
| 2 | [1.1898, 1.9073] | 9999 | 0.04985 |
| 3 | [1.9073, 2.6149] | 10000 | 0.04993 |
| 4 | [2.6149, 3.3153] | 9999 | 0.05002 |
| 5 | [3.3153, 4.0905] | 10000 | 0.04989 |
| 6 | [4.0905, 4.8748] | 10000 | 0.04961 |
| 7 | [4.8748, 6.0122] | 9999 | 0.05002 |
| 8 | [6.0122, 7.0513] | 10000 | 0.05069 |
| 9 | [7.0513, 9.6228] | 10000 | 0.05023 |

Flat: residual standard deviation is essentially the same in every bin, equal to σ = 0.05. **The additive coordinate has homoskedastic residuals; the multiplicative coordinate does not.**

## What this experiment establishes

The substrate's jet-SPTLS machinery is **correct only in the coordinate where the trajectory's underlying operation linearises.** For the log-normal walk:

  - on the **log** side (additive coordinate), the machinery reads the operator cleanly, the planted random-walk M̂ comes out, residuals are white with stationary variance σ² — exactly E17 in another scale;

  - on the **multiplicative** side, the machinery reads a level-row that *looks* like the random-walk operator to first order (because S_{t+1} ≈ S_t · (1 + X) for small X), but the residual scale grows linearly with the level. The variance of the residual is not a single number; it is S_t² · σ². The substrate's residual summary is mis-applied unless we factor out the level-dependence.

**The substrate's choice of coordinate is part of its reading.** For multiplicative processes the coordinate is the logarithm; for additive ones it is the trajectory itself. The substrate machinery is invariant to this choice only if you remember to make it.

This closes the qualitative gap we observed in E09: the log-normal's classical moments grow exponentially with horizon and become operationally useless, but the substrate sees a Gaussian random walk on the log side, which is a clean, well-behaved process. The two sides are connected by the choice of coordinate, and the substrate's job is to make that choice explicit — not to hide it.
