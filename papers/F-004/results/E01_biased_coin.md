# E01 — The biased coin trajectory  (p = 0.3)

**Sample length:** N = 10000.  **Probability of heads:** p = 0.3.  **Closed-form marginal at each horizon:** P(S_n = k) = C(n, k) · p^k · (1−p)^(n−k)  (the same path-count expression as for the fair coin, with the paths now weighted by the probability of their head-tail composition).

## The marginal is no longer symmetric at finite horizons

When the coin is fair the marginal at any horizon is left-right symmetric: P(S_n = k) = P(S_n = n − k). With p = 0.3 that mirror symmetry breaks. The marginal is still concentrated around its mean (now n·p, not n/2), its spread is still √(n·p·(1−p)), but the right tail is shorter than the left tail at every finite n.

The asymmetry of the standardised marginal (S_n − n·p) / √(n·p·(1−p)) shrinks as the horizon grows. We report two cross-checks of this at each horizon: the closed-form skew (1−2p)/√(n·p·(1−p)) and a direct calculation of the same skew from the path-count expression itself. They agree.

| horizon n | skew of standardised marginal (closed form) | cross-check from path-count |
| ---: | ---: | ---: |
| 5 | +0.3904 | +0.3904 |
| 10 | +0.2760 | +0.2760 |
| 50 | +0.1234 | +0.1234 |
| 500 | +0.0390 | +0.0390 |

The asymmetry decays roughly as 1/√n. At horizon 500 the standardised marginal is nearly symmetric.

## Empirical marginals vs path-count

| horizon n | total-variation distance (empirical vs path-count) | n_reps |
| ---: | ---: | ---: |
| 5 | 0.0019 | 200000 |
| 10 | 0.0023 | 200000 |
| 50 | 0.0033 | 200000 |
| 500 | 0.0065 | 200000 |

## The substrate's view of this trajectory

The trajectory has no memory: each flip is independent of the past. The one-step operator on the jet is the cumulative identity. **All the substantive content of the trajectory lives in the step itself**, which is now asymmetric.

Reading the step (centred to mean zero) from the sample:

| quantity | value | expected |
| :--- | ---: | ---: |
| support of the step                  | [-0.3, 0.7] | (−p, 1−p) = (-0.3, 0.7) |
| empirical probabilities              | [0.6992, 0.3008] | (0.7, 0.3) |
| mean                                  | +0.000800 | 0 |
| variance                              | 0.210319 | 0.210000 |
| third moment                          | +0.084296 | +0.084000 |
| fourth moment                         | 0.077886 | — |

**The third moment of the step is non-zero**, which is the first departure from E00. It is exactly p·(1−p)·(1−2p), and it carries the asymmetry of the marginal at finite horizons through the convolution.

## The marginal at horizon n is the n-fold convolution of the step

Computing the n-fold self-convolution of the step distribution and comparing to the closed-form path-count expression at each horizon, with two versions of the step: the exact (0.7, 0.3) measure and the empirical measure read from this particular sample (0.699 / 0.301).

| horizon n | L¹ (exact-step convolution vs path-count) | L¹ (empirical-step convolution vs path-count) |
| ---: | ---: | ---: |
| 5 | 7.37e-18 | 3.29e-03 |
| 10 | 1.27e-16 | 4.27e-03 |
| 50 | 3.98e-16 | 9.79e-03 |
| 500 | 3.00e-16 | 3.11e-02 |

With the exact step the convolution matches the path-count expression to floating-point precision: the marginal at any horizon is, *exactly*, the n-fold convolution of the single-step measure. The asymmetry of the step is carried faithfully through the convolution into the asymmetry of the marginal at finite n.

## What this experiment shows

The whole story of E00 carries over with one new ingredient. The step now has a non-zero third moment; the marginal at finite horizons is visibly asymmetric; the asymmetry of the standardised marginal decays as 1/√n; the convolution identity still holds exactly. The smooth bell still appears at large enough n — but the path to it passes through an asymmetric regime that the symmetric coin did not have to traverse.
