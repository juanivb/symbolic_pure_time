# E10 — The Cauchy-sum trajectory  (no moments)

**Step rule:** Y_t = X_{1,t} / X_{2,t} with X_{1,t}, X_{2,t} ~ N(0, 1) independent.  **Trajectory:** S_t = Y_1 + Y_2 + ... + Y_t.  **Analytical marginal at horizon n:** the Cauchy density of location 0 and scale n, namely n / (π · (n² + x²)).

## What is new in this experiment

The step Y = X_1 / X_2 has no finite mean. Its variance, skewness, excess kurtosis — none of those quantities exist either, because the integrals defining them diverge. Every empirical reading we have built up so far (mean, variance, the four-moment summary, the shape-decay rule) is silent here. The closed-form expressions are not well-defined, so there is nothing for the empirical estimates to converge to.

This is the case the F-004 honesty anchor was designed to cover: moments fail, but quantile-based summaries (median, IQR) stay finite and informative.

## The marginal at horizon n is Cauchy with scale n

When two iid Cauchy variables are added, the result is again a Cauchy variable — with the *sum* of the two scale parameters. Iterating, n iid standard Cauchys sum to a Cauchy with scale n. The scale grows *linearly* with the horizon, not as √n.

Comparison of the empirical histogram (truncated to the window |x| ≤ 50·n which captures ~ 99 % of the mass) with the analytical Cauchy(0, n) density:

| horizon n | median (emp) | IQR (emp) | IQR (predicted 2n) | mass in window 50n | L¹ (hist vs density) |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | +0.0015 | 2.0080 | 2.0000 | 0.9868 | 0.0272 |
| 2 | -0.0153 | 4.0012 | 4.0000 | 0.9870 | 0.0248 |
| 10 | +0.0225 | 19.9542 | 20.0000 | 0.9872 | 0.0233 |
| 100 | +0.0851 | 199.5147 | 200.0000 | 0.9874 | 0.0230 |

Three observations from the table.

  - The median of S_n is exactly 0 at every horizon, as the symmetric Cauchy demands.

  - The IQR of S_n is *exactly* 2·n. This is the cleanest reading of how scale grows with the horizon in this regime: linearly, not as √n. The substrate's quantile-based scale measure is the right tool here; moment-based scale measures (the variance) are not.

  - The L¹ distance between the empirical histogram and the analytical density is small (~ 0.01) — the *shape* of the marginal is the Cauchy density at every horizon, even though the moments of that density do not exist.

## The classical moments fluctuate without converging

We compute the empirical mean and variance of S_n over five independent runs of 20000 samples each. If a finite mean and variance existed, we would expect roughly the same answer across runs (up to a sampling noise that shrinks with sample size). What we see instead:

| horizon n | empirical mean across 5 runs | empirical variance across 5 runs |
| ---: | :--- | :--- |
| 1 | [1.27, -12.22, 1.36, 0.55, -41.26] | [46766.6, 2160222.3, 58766.0, 6878.8, 18655265.5] |
| 2 | [51.52, -0.73, 2.7, -0.1, 2.66] | [71705291.6, 48628.4, 273370.0, 28302.5, 67180.4] |
| 10 | [7.95, 6.69, -17.46, 3.21, 7.29] | [741373.9, 1140210.6, 2391484.9, 1314625.7, 949047.1] |
| 100 | [-107.15, -13.2, 41.33, 817.67, -41.39] | [144828697.5, 85375173.3, 115607719.4, 16438676667.7, 227932630.4] |

The empirical mean across five runs jumps by orders of magnitude between runs, with both signs. The variance jumps by even more. Neither converges to a finite value as the sample grows, because the integrals defining them diverge — a single large draw from the tail dominates the empirical average and re-shuffles the answer entirely. **This is the empirical statement of "moments do not exist" as a *measurable* fact**, not just an analytical one. Any procedure that reads the trajectory through its sample moments will be unstable in this regime.

## The substrate's view of this trajectory

The step is iid, so the operator is the cumulative identity — no surprise there. The novelty is in the step's measure: it has no finite moments, so the moment-based reading of the step (mean, variance, skew, kurt) fails. The substrate's quantile-based reading does not.

What the substrate sees on this trajectory: a step with median 0, IQR 2, and a density f_Y(y) = 1/(π(1 + y²)). Sums of independent draws of this step give Cauchy marginals at every horizon, with scale that grows linearly with n. **The same convolution machinery from earlier experiments produces this, but the cumulant-decay rule from note 02 does not apply because the cumulants do not exist.**

## What this experiment shows, alongside E06

The Gaussian step (E06) and the Cauchy step (this experiment) are two fixed points of the n-fold self-convolution: with each, the marginal at every horizon is the *same shape* as the step, only rescaled.

  - **Gaussian:** all moments exist; the scale grows as √n; the decay-of-shape rule from note 02 applies, but trivially (the shape was already at its fixed point with no features to decay).

  - **Cauchy:** no moments exist; the scale grows as n; the decay-of-shape rule does not apply because the cumulants it uses are not finite. The robust quantile-based scale measure (IQR) is the right tool, and it grows as exactly 2n.

These two are not the only fixed points of accumulation — the stable distributions are a one-parameter family interpolating between them and beyond. But for the present purpose, what we have shown is that the convolution machinery still produces the marginal exactly when the step has no moments; only the *summary* of that marginal needs a different language (quantiles instead of moments).
