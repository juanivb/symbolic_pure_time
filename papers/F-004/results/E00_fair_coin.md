# E00 — The fair coin trajectory

**Sample length:** N = 10000.  **Closed-form marginal at each horizon:** P(S_n = k) = C(n, k) / 2^n  (the number of paths to height k in n steps, divided by the total number of paths).

## Empirical marginals vs path-count

| horizon n | total-variation distance (empirical vs path-count) | n_reps |
| ---: | ---: | ---: |
| 5 | 0.0012 | 200000 |
| 10 | 0.0027 | 200000 |
| 50 | 0.0040 | 200000 |
| 500 | 0.0066 | 200000 |

## The marginal at n = 500 versus a smooth bell of the same width

At horizon n = 500 the discrete marginal P(S_n = k) and the smooth bell of mean n/2 and spread √(n/4) sit on top of each other.

**Total L¹ error between the two = 0.0005**  (both summed across k = 0..n).

## The substrate's view of this trajectory

The trajectory has no memory: each step is independent of the past. The one-step operator on the jet is therefore the cumulative identity — the next level is the previous level plus the new step. There is no rotation, no anisotropic stretch, no acceleration. **All the substantive content of the trajectory lives in the step itself.**

Reading the step (centred to mean zero) from the sample:

| quantity | value |
| :--- | ---: |
| support of the step                  | [-0.5, 0.5]  (expected ±½) |
| empirical probabilities              | [0.497, 0.503]  (expected ½, ½) |
| mean                                  | +0.003000  (expected 0) |
| variance                              | 0.249991  (expected 0.25) |
| third moment (skew, raw)              | +0.000750  (expected 0) |
| fourth moment                         | 0.062500  (expected 1/16 = 0.0625) |

## The marginal at horizon n is the n-fold convolution of the step

Computing the n-fold self-convolution of the step distribution and comparing to the closed-form path-count expression at each horizon, with two versions of the step: the exact ½/½ measure and the empirical measure read from this particular sample (0.497 / 0.503).

| horizon n | L¹ (exact-step convolution vs path-count) | L¹ (empirical-step convolution vs path-count) |
| ---: | ---: | ---: |
| 5 | 0.00e+00 | 1.12e-02 |
| 10 | 0.00e+00 | 1.48e-02 |
| 50 | 0.00e+00 | 3.38e-02 |
| 500 | 2.54e-16 | 1.07e-01 |

With the exact step the convolution matches the path-count expression to floating-point precision: the marginal at any horizon is, *exactly*, the n-fold convolution of the single-step measure. With the empirical step (which has a small sampling bias of ~0.003 in its probabilities), the discrepancy grows roughly linearly with the horizon, because the bias compounds n times — and that is itself a clean check on the convolution identity.
