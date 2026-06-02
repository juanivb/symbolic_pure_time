# E14 — The symmetric α-stable sum  (α = 1.5)

**Step:** symmetric α-stable, α = 1.5, scale 1.  **Trajectory:** S_n = Y_1 + ... + Y_n with Y_t iid.  **Defining property:** the sum of n iid copies is again α-stable with scale γ · n^(1/α). For α = 1.5 the scale grows as **n^(2/3)**.

## Where this experiment sits

Together with E06 (Gaussian, α = 2) and E10 (Cauchy, α = 1) this closes the family of convolution-fixed-points the programme has encountered:

| α | family | scale of S_n | finite moments | examples in this programme |
| ---: | :--- | :--- | :--- | :--- |
| 2.0 | Gaussian | √n | all | E06 |
| 1.5 | symmetric α-stable | n^(1/1.5) = n^(0.6667) | order < 1.5 only | E14 (this) |
| 1.0 | Cauchy | n | none | E10 |

The family of symmetric α-stable distributions is a one-parameter interpolation between these two: every α ∈ (0, 2] gives a fixed point of self-convolution with scale n^(1/α). The two we have met before (α = 1 and α = 2) are the endpoints; α = 1.5 sits in the middle, with mean defined and variance infinite.

## The step

Generated via the Chambers–Mallows–Stuck construction: take U uniform on (−π/2, π/2) and W exponential with mean 1 independent, and combine them in closed form. The step has

  - mean = 0 by symmetry (empirically median ≈ 0,    sample IQR ≈ 1.93);
  - all moments of order ≥ α = 1.5 infinite.

## The IQR scales as n^(1/α) = n^(2/3)

Comparing the empirical IQR of S_n at each horizon against three candidate scaling laws: the α-stable prediction n^(2/3), the Gaussian-style prediction n^(1/2), and the Cauchy-style prediction n.

| horizon n | median | IQR (emp) | IQR (n^(2/3)) | IQR if Gaussian (√n) | IQR if Cauchy (n) |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | -0.0012 | 1.9359 | 1.9304 | 1.9304 | 1.9304 |
| 10 | -0.0197 | 8.9684 | 8.9603 | 6.1046 | 19.3044 |
| 100 | -0.0134 | 41.6927 | 41.5901 | 19.3044 | 193.0443 |
| 1000 | -0.5417 | 194.1589 | 193.0443 | 61.0460 | 1930.4432 |

The empirical IQR follows the n^(2/3) law cleanly across the four horizons. The Gaussian-style √n prediction under-estimates the spread (the step is heavier-tailed than Gaussian, so its sum spreads faster than √n); the Cauchy-style n prediction over-estimates (the step is lighter than Cauchy, so its sum spreads slower than linearly). The α-stable scaling is the right interpolation.

## Empirical variance fluctuates without converging

Because the variance integral diverges, no finite-sample estimate converges to anything. Five runs of 20000 samples each:

| horizon n | variance across 5 runs |
| ---: | :--- |
| 1 | [37.8, 35.3, 38.3, 630.1, 38.8] |
| 10 | [994.4, 429.3, 1130.3, 858.2, 519.2] |
| 100 | [37116.8, 23847.6, 9763.6, 19793.4, 47813.1] |
| 1000 | [182000.5, 614637.3, 176926.7, 338576.7, 204808.1] |

Variance estimates can vary by orders of magnitude between runs, and there is no convergence: the empirical variance of S_n is dominated by the single largest sample, which can be arbitrarily large with positive probability. **Variance is not a meaningful scale measure for this process.** The IQR is.

## Shape invariance: S_n / n^(2/3) recovers the step

If S_n is α-stable with scale n^(2/3), then S_n / n^(2/3) is α-stable with scale 1 — that is, the same distribution as the step itself. We compare the empirical histogram of (S_n / n^(2/3)) against the empirical histogram of the step, both restricted to the body |x| ≤ 5:

| horizon n | scale used (n^(2/3)) | L¹ (histogram(S_n / scale) vs histogram(step)) |
| ---: | ---: | ---: |
| 1 | 1.0000 | 0.0168 |
| 10 | 4.6416 | 0.0205 |
| 100 | 21.5443 | 0.0212 |
| 1000 | 100.0000 | 0.0146 |

The L¹ distance between the standardised histogram and the step histogram is at sampling-noise level (around 0.03) at every horizon. The shape is invariant — the marginal at any horizon is the same α-stable shape as the step, only rescaled by n^(2/3).

## The substrate's view of this trajectory

Same as E10: the operator is the cumulative identity, all content lives in the step, and the step has no finite variance — so all moment-based summaries fail. The quantile-based summary (median and IQR) is finite and informative. The new piece relative to E10 is that the IQR scales as n^(2/3), not n: the substrate's scale-measure carries the tail-thickness of the step through the exponent of n.

## What this experiment shows, alongside E06 and E10

There is not one fixed point of self-convolution; there is a one-parameter family. Each α ∈ (0, 2] gives a stable family whose self-convolution preserves shape and scales as n^(1/α). The Gaussian is the α = 2 case (lightest tail among the family); the Cauchy is the α = 1 case; and the intermediate cases (α ∈ (1, 2)) sit between, with mean defined and variance infinite.

Most heavy-tailed real-world data — financial returns, network flow distributions, earthquake magnitudes — has been measured to have effective α in this intermediate range (between roughly 1.3 and 1.8). The substrate's machinery handles them all with the same machinery: identify the step, measure its IQR-scaling exponent, propagate it through the convolution.
