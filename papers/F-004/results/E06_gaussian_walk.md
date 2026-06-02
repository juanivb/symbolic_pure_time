# E06 — The Gaussian random walk trajectory

**Step distribution:** X_t ~ N(0, 1), independent across time.  **Trajectory:** S_t = X_1 + ... + X_t.  **Analytical marginal at horizon n:** the bell-shaped density of mean 0 and variance n.

## What is new in this experiment

The step is now continuous (a real number drawn from a smooth bell) rather than discrete. Three things follow:

  - the marginal at horizon n is itself a smooth density, not a list of probabilities at integer values;
  - the convolution identity from E00 – E05 still applies, but the convolution is now an integral over a continuous variable rather than a sum over integers;
  - the Gaussian step has the property that the n-fold convolution of itself is *again* a Gaussian of width √n. This is true *exactly*, at every horizon, not asymptotically. Of all the step shapes we have studied, the Gaussian is the only one whose marginal at finite horizons sits on the same family as the limit shape that emerged at large horizons for every other step.

## Empirical marginals vs analytical Gaussian density

At each horizon n we sample 200000 fresh draws of S_n, build a normalised histogram of those draws on 80 bins, and compare to the analytical density of N(0, n) evaluated at the same bin centres. The L¹ distance between the two normalised densities (weighted by the bin width) is a direct read of how far the empirical histogram sits from the analytical bell:

| horizon n | empirical variance | predicted variance | empirical skew | empirical excess kurt | L¹ (hist vs density) |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1 |    1.0032 | 1.0 | -0.0056 | -0.0027 | 0.0104 |
| 10 |   10.0250 | 10.0 | +0.0031 | -0.0009 | 0.0111 |
| 100 |   99.9594 | 100.0 | -0.0028 | -0.0053 | 0.0114 |
| 1000 | 1000.3709 | 1000.0 | -0.0008 | -0.0031 | 0.0131 |

Three observations from the table:

  - the variance of the marginal grows linearly with n, matching the prediction Var(S_n) = n to sampling noise at every horizon;

  - the standardised skew and excess kurtosis stay near zero at every horizon — the step already starts at zero for both, so there is no shape feature for the convolution to wear down;

  - the L¹ distance between the empirical histogram and the analytical N(0, n) density is at sampling-noise level (around 0.03 – 0.05 on 80 bins with 200 000 samples). The discrete histogram is a finite-sample approximation of the same smooth density that the analytical formula gives.

## The substrate's view of this trajectory

Same as in E00 – E03: the step is independent of the past, so the one-step operator is the cumulative identity. **All substantive content lives in the step**, which is now a continuous Gaussian instead of a discrete two-point measure. The step's complete description is two numbers: mean 0 and variance 1. The marginal at any horizon is determined by these two numbers alone, via Var(S_n) = n, mean(S_n) = 0, and the shape is the same bell.

## What this experiment shows

The convolution identity that ran through E00 – E03 (in the iid regime) carries over to the continuous step without modification. The Gaussian step is a *fixed point* of self-convolution up to rescaling — convolving N(0, 1) with itself produces N(0, 2), which after rescaling is N(0, 1) again. This is the shape that the long-horizon bell of every previous experiment was converging to. With the Gaussian step it appears at *every* horizon, not just at large ones.

This experiment closes the iid arc of the programme. The next experiment (E07) starts from this Gaussian step and constructs a derived continuous distribution — the sum of squared Gaussians — which gives us the first non-Gaussian continuous shape obtained by accumulating a transformed Gaussian step.
