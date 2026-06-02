# E07 — The sum-of-squared-Gaussians trajectory

**Step rule:** draw X_t ~ N(0, 1) independently across t; set Y_t = X_t².  **Trajectory:** S_t = Y_1 + ... + Y_t.  **Analytical marginal at horizon n:** the χ²(n) density x^(n/2 − 1) · exp(−x/2) / (2^(n/2) · Γ(n/2)) for x ≥ 0.

## What is new in this experiment

Up to E06 the step was either symmetric (E00, E02, E06) or asymmetric but discrete (E01, E03). Here the step is *continuous and asymmetric*: Y = X² is strictly positive, has a single peak near zero and a long right tail. Sums of such steps produce the χ²(n) family — a continuous distribution that emerges by *transforming* the standard Gaussian step and then accumulating. It is the first non-Gaussian continuous distribution in the programme, and it is built entirely from operations applied to the simple Gaussian step.

## The step Y = X² and its four moments

Squaring a standard Gaussian gives a distribution with specific moments that we read directly from the sample:

| quantity | empirical | expected |
| :--- | ---: | ---: |
| mean             | 1.0024 | 1 |
| variance         | 2.0176 | 2 |
| skewness         | 2.8543 | √8 ≈ 2.8284 |
| excess kurtosis  | 12.2546 | 12 |

These four numbers determine, by the convolution identity and the cumulant additivity, the four leading moments of the marginal at any horizon.

## Empirical marginal vs analytical χ²(n)

At each horizon n we draw 200000 fresh trajectories, accumulate n squared Gaussian steps in each, and compare the resulting histogram against the χ²(n) density evaluated at the same bin centres.

| horizon n | mean (emp) | mean (n) | var (emp) | var (2n) | L¹ (hist vs density) |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 1.0032 | 1 | 2.0021 | 2 | 0.1224 |
| 2 | 1.9980 | 2 | 3.9976 | 4 | 0.0088 |
| 10 | 9.9916 | 10 | 20.0487 | 20 | 0.0099 |
| 100 | 99.9694 | 100 | 199.9244 | 200 | 0.0112 |

Mean grows as n (cumulant additivity), variance grows as 2n (twice the step variance), and the empirical histogram sits on the analytical density at sampling-noise level (L¹ around 0.01).

## Shape decay (the cumulant rule from note 02 in action)

Note 02 established that the m-th standardised cumulant of the marginal decays as n^(1 − m/2). For this step:

  - the step has skewness √8 ≈ 2.83, so the marginal at horizon n should have skewness √(8/n) — a 1/√n decay scaled by the step's own skewness;
  - the step has excess kurtosis 12, so the marginal should have excess kurtosis 12/n — a 1/n decay.

| horizon n | skew (emp) | skew (√(8/n)) | excess kurt (emp) | excess kurt (12/n) |
| ---: | ---: | ---: | ---: | ---: |
| 1 | +2.8032 | +2.8284 | +11.5645 | +12.0000 |
| 2 | +2.0039 | +2.0000 | +5.9680 | +6.0000 |
| 10 | +0.9111 | +0.8944 | +1.2776 | +1.2000 |
| 100 | +0.2751 | +0.2828 | +0.1035 | +0.1200 |

Empirical values match the predicted decays at every horizon. At n = 1 the marginal has skewness 2.83 and excess kurtosis 12 — the whole asymmetry of the step. At n = 100 those have shrunk to ≈ 0.28 and ≈ 0.12 respectively, two orders of magnitude lower, and the marginal is starting to look bell-shaped. This is the decay-of-shape rule operating on a continuous, asymmetric step.

## The substrate's view of this trajectory

Same as in E00 – E03 and E06: the step is independent of the past, so the one-step operator is the cumulative identity. All substantive content lives in the single-step density f_Y(y), which is itself the squared-Gaussian density 1/√(2π y) · exp(−y/2) for y > 0. The marginal at any horizon is the n-fold self-convolution of this density, which has a closed form (the χ²(n) family above) but the convolution operation itself is doing all the work.

## What this experiment shows

The convolution identity carries over without modification to a continuous, asymmetric step. The χ² family at any number of degrees of freedom emerges as the n-fold self-convolution of the squared-Gaussian step. The shape-decay rule from note 02 controls the rate at which the chi-squared marginal becomes bell-shaped at large n: skewness shrinks as √(8/n), excess kurtosis as 12/n. The machinery is exactly the same as in the earlier experiments — only the step has changed.

Together with E06 (the un-transformed Gaussian walk gives the Gaussian family) this shows that two of the classical continuous distributions of probability are accumulated trajectories of a single Gaussian step: one direct (Gaussian) and one squared (chi-squared). Other classical continuous distributions follow by applying further simple operations to the same trajectory.
