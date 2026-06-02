# E09 — The log-normal trajectory (running product)

**Step rule:** M_t = exp(X_t) with X_t ~ N(0, 1) independent across t.  **Trajectory (multiplicative side):** S_t = M_1 · M_2 · … · M_t.  **Trajectory (additive side):** W_t = log S_t = X_1 + X_2 + … + X_t  (the Gaussian walk of E06).  **Analytical marginal at horizon n:** S_n has the log-normal density with parameters (μ = 0, σ² = n); W_n has the Gaussian density N(0, n).

## What is new in this experiment

All earlier experiments accumulated steps additively. Here the accumulation is multiplicative — the state at step t is a product, not a sum. This is the first qualitative change to the accumulation machinery in the programme. The change is, however, less mysterious than it sounds: the logarithm of the trajectory is exactly the additive Gaussian walk we already know from E06. So everything that happens here is the Gaussian walk seen through an exponential lens, and the new ingredients show up on the *multiplicative* side as an unbounded amplification of moments.

## The two sides at each horizon

Reading both the multiplicative S_n and the additive W_n = log S_n from the same trajectories:

**Multiplicative side** — the moments of S_n grow with n; the mean grows as exp(n/2), the variance grows as exp(2n) − exp(n), the skewness as (e^n + 2)·√(e^n − 1).

| horizon n | mean (emp) | mean (pred) | var (emp) | var (pred) | skew (emp) | skew (pred) |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 1.65 | 1.6487 | 4.73e+00 | 4.6708e+00 | 6.30 | 6.18e+00 |
| 2 | 2.72 | 2.7183 | 4.57e+01 | 4.7209e+01 | 17.90 | 2.37e+01 |
| 10 | 144.06 | 148.4132 | 2.65e+07 | 4.8514e+08 | 151.49 | 3.27e+06 |
| 100 | 510355823632676.88 | 5184705528587072045056.0000 | 5.07e+34 | 7.2260e+86 | 447.20 | 1.39e+65 |

Even at moderate horizons (n = 100), the predicted skewness of the log-normal marginal is astronomical. The empirical estimate from 200 000 samples does not match these predictions for large n: the tail of the log-normal at large σ² is so extreme that any finite sample under-represents it, and the empirical skewness is dominated by the largest few samples. **This is itself the qualitative finding:** on the multiplicative side, the shape-decay rule from note 02 does *not* apply — every horizon brings more extreme tails, not less.

**Additive side** — taking the logarithm of the trajectory recovers the Gaussian walk of E06: the variance of W_n grows linearly with n, skew and excess kurtosis stay near zero, and the shape-decay rule applies in its trivial form (the step had no shape to decay to begin with).

| horizon n | mean (emp) | mean (pred) | var (emp) | var (pred) | skew (emp) | excess kurt (emp) |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | +0.0001 | 0.0 | 1.0024 | 1 | -0.0036 | +0.0079 |
| 2 | +0.0044 | 0.0 | 2.0004 | 2 | -0.0048 | -0.0071 |
| 10 | -0.0017 | 0.0 | 10.0004 | 10 | -0.0002 | -0.0090 |
| 100 | +0.0068 | 0.0 | 99.8444 | 100 | +0.0052 | +0.0168 |

Perfectly tame, as expected — the additive side is E06 by another name.

## Empirical histogram vs analytical log-normal density

On the multiplicative side, we compare the histogram of S_n (in log-spaced bins, to handle the heavy right tail) against the log-normal density with parameters (0, n). The L¹ distance between the two normalised densities:

| horizon n | L¹ (hist vs density, multiplicative side) |
| ---: | ---: |
| 1 | 0.0111 |
| 2 | 0.0125 |
| 10 | 0.0124 |
| 100 | 0.0340 |

The histogram tracks the density to sampling-noise level even though the moments diverge from their predicted values for large n. The *shape* of the marginal is correct; the moment-based summary just fails to describe it once the tail is heavy enough that finite samples cannot resolve it.

## The substrate's view of this trajectory

Two equivalent readings, depending on which side we look from:

  - **Additive side:** the operator is the cumulative identity, the step is a Gaussian, the marginal at horizon n is N(0, n). Same as E06 exactly.

  - **Multiplicative side:** the operator multiplies by the next M_t. Reading at the log scale linearises this to the additive operator above. The marginal at horizon n on this side is log-normal, and its moments grow with n rather than shrink — but the *shape* in log-space is exactly the Gaussian walk and follows the additive rules.

## What this experiment shows

When the accumulation is multiplicative, the underlying additive structure lives in the logarithm. The classical log-normal distribution at horizon n is exp(W_n) where W_n is the Gaussian walk after n steps. So far the convolution identity has been the single source of every marginal we computed; here we see its first generalisation: when the operation is product rather than sum, the same identity applies on the log scale.

The shape-decay rule from note 02 still controls the additive side. On the multiplicative side, the corresponding statement is a *growth* of shape — moments of S_n inflate with n at exponential rates, not because the rule fails but because the rule is the wrong invariant for this operation. The right invariant lives in the logarithm.
