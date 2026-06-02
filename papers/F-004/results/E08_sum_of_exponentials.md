# E08 — The sum-of-exponentials trajectory

**Step rule:** Y_t ~ Exp(1) independent across t.  **Trajectory:** S_t = Y_1 + ... + Y_t.  **Analytical marginal at horizon n:** the Gamma(n, 1) density x^(n − 1) · exp(−x) / Γ(n) for x ≥ 0.

## What is new in this experiment

A third continuous, asymmetric, positive-support step — the exponential. The accumulation of n such steps gives the Gamma family at integer shape parameter, which together with E06 (Gaussian) and E07 (chi-squared) gives us three of the classical continuous distributions of probability as direct outcomes of the same accumulation machinery.

## The step Y ~ Exp(1) and its four moments

Reading the four leading moments of the step directly from the sample:

| quantity | empirical | expected |
| :--- | ---: | ---: |
| mean             | 0.9986 | 1 |
| variance         | 1.0047 | 1 |
| skewness         | 2.0107 | 2 |
| excess kurtosis  | 6.0454 | 6 |

## Empirical marginal vs analytical Gamma(n, 1)

At each horizon n we draw 200000 fresh trajectories, accumulate n exponential steps in each, and compare the resulting histogram against the Gamma(n, 1) density.

| horizon n | mean (emp) | mean (n) | var (emp) | var (n) | L¹ (hist vs density) |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 1.0018 | 1 | 1.0126 | 1 | 0.0095 |
| 2 | 1.9966 | 2 | 1.9965 | 2 | 0.0097 |
| 10 | 9.9978 | 10 | 9.9363 | 10 | 0.0129 |
| 100 | 99.9954 | 100 | 100.3036 | 100 | 0.0133 |

Mean and variance both grow as n (the step has mean 1 and variance 1, and the convolution identity adds them across n steps). The empirical histogram sits on the analytical density at sampling-noise level.

## Shape decay

The step has skewness 2 and excess kurtosis 6. By the rule from note 02, the marginal at horizon n should have skewness 2/√n and excess kurtosis 6/n.

| horizon n | skew (emp) | skew (2/√n) | excess kurt (emp) | excess kurt (6/n) |
| ---: | ---: | ---: | ---: | ---: |
| 1 | +2.0217 | +2.0000 | +6.1117 | +6.0000 |
| 2 | +1.4150 | +1.4142 | +2.9750 | +3.0000 |
| 10 | +0.6212 | +0.6325 | +0.5722 | +0.6000 |
| 100 | +0.2062 | +0.2000 | +0.0558 | +0.0600 |

At n = 1 the marginal is the step itself, with skewness 2 and excess kurtosis 6. At n = 100 those have shrunk to ≈ 0.2 and ≈ 0.06 respectively — the Gamma(n, 1) marginal is now nearly bell-shaped. The same rule applies as in E01, E02, E07: each feature of the step decays at its own 1/n^(m/2 − 1) rate, and the leading constants 2 and 6 are the only specific marks the exponential step leaves.

## The substrate's view of this trajectory

Independent step → operator is the cumulative identity. All content lives in the step density f_Y(y) = exp(−y) on [0, ∞). The marginal at any horizon is the n-fold self-convolution of this density, with closed form Gamma(n, 1).

## What this experiment shows

Three continuous distributions of probability now appear as outcomes of the same accumulation, applied to three steps:

  - E06: Gaussian step                  → Gaussian marginal at any horizon.
  - E07: Gaussian step, squared          → chi-squared marginal.
  - E08: exponential step                → Gamma marginal.

Each step has its own four-moment fingerprint, and the marginal at horizon n carries that fingerprint scaled by the n^(1 − m/2) decay. Nothing is postulated; everything follows from the step.
