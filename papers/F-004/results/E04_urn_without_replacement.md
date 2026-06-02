# E04 — The urn-without-replacement trajectory

**Urn:** 50 red balls, 50 blue balls, total N = 100.  **Trajectory:** draw all N balls in random order without replacement; X_t = 1 if the t-th draw is red, 0 otherwise.  **State at step t:** S_t = count of reds drawn so far.

## What is new in this experiment

For the first time the step is *not* independent of the past. Once a red ball is drawn there are fewer reds in the urn, so the probability of red on the next draw goes down. The step itself depends on the state of the urn, which in turn depends on the history of draws.

This breaks the central identity of E00–E03 — the one that says *marginal at horizon n = n-fold self-convolution of the single step*. The break is *quantitative*: we know exactly how much it differs, and the difference is the finite-population correction factor.

## Empirical marginal vs the two candidate closed forms

At each horizon we compare the empirical marginal (from 100000 random draws of the urn) against two closed-form predictions:

  - **without-replacement path-count** P(S_n = k) = C(R, k) · C(B, n − k) / C(N, n) — counts the number of ways to choose k reds out of R and n − k blues out of B, divided by the total number of n-subsets;

  - **iid path-count** C(n, k) · p^k · (1 − p)^(n − k) with p = R/N = 0.5 — the answer we would get if each draw were independent (the regime of E00–E03).

| horizon n | TV(emp, without-replacement) | TV(emp, iid) | TV(without-replacement, iid) |
| ---: | ---: | ---: | ---: |
| 5 | 0.0014 | 0.0130 | 0.0128 |
| 10 | 0.0032 | 0.0255 | 0.0259 |
| 50 | 0.0033 | 0.1634 | 0.1625 |
| 99 | 0.0006 | 0.8408 | 0.8408 |

Reading the columns: the empirical marginal sits on the without-replacement curve to within sampling noise; it does *not* sit on the iid curve. The gap between the two closed forms grows with n, with the largest gap near n = N/2 (when the urn is half drained and the conditional probability has drifted the most).

## The variance shrinks by a known finite-population factor

If the draws were independent, the variance of S_n would be n · p · (1 − p) with p = 0.5. With the without-replacement rule, the variance is smaller by a factor (N − n) / (N − 1). Measured directly:

| horizon n | var (empirical) | var (without-replacement closed form) | var (iid prediction) | var (without-replacement prediction) | correction (N−n)/(N−1) |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 5 | 1.1967 | 1.1995 | 1.2500 | 1.1995 | 0.9596 |
| 10 | 2.2712 | 2.2727 | 2.5000 | 2.2727 | 0.9091 |
| 50 | 6.2757 | 6.3131 | 12.5000 | 6.3131 | 0.5051 |
| 99 | 0.2500 | 0.2500 | 24.7500 | 0.2500 | 0.0101 |

At n = 99 the correction factor is (100 − 99)/99 ≈ 0.0101 and the variance is about 100× smaller than the iid prediction — by then we have drawn the whole urn and almost all randomness is gone (only the order is random; the count of reds at the end is exactly R).

## The substrate's view: the step has a non-trivial state-dependence

For each time t we measure two things across 20000 trajectories:

  - the **unconditional** probability that the t-th draw is red, which should be exactly R/N = 0.5 for every t (the symmetry of the urn);
  - the **trajectory-averaged conditional prediction** (R − S_{t−1})/(N − (t − 1)), which is the substrate's one-step operator reading at time t: given the history, this is the probability of red on the next draw.

A few representative time points:

| t | P(X_t = red) empirical | conditional prediction averaged over trajectories | mean squared residual |
| ---: | ---: | ---: | ---: |
| 1 | 0.4992 | 0.5000 | 0.2500 |
| 10 | 0.4961 | 0.4999 | 0.2499 |
| 50 | 0.4985 | 0.4995 | 0.2479 |
| 80 | 0.5000 | 0.4993 | 0.2406 |
| 90 | 0.4985 | 0.4990 | 0.2295 |
| 99 | 0.4991 | 0.4991 | 0.1269 |

Two things to notice. The unconditional probability is essentially R/N = 0.5 at every t (modulo sampling noise) — the urn is *symmetric* under permutation of draw order. The conditional prediction averaged over trajectories also reads R/N at every t, for the same symmetry reason. But the mean squared residual (how far X_t lies from the conditional prediction *within a single trajectory*) shrinks as t grows: as the urn drains, the conditional probability becomes more and more determined by the past, until near t = N the next draw is essentially forced by what is left.

This is the substrate's first genuine job in the programme: tracking the conditional probability of the next step given the history. In E00–E03 that conditional probability was constant and equal to the unconditional one — the operator was the identity. Here the conditional probability is a non-trivial function of the past, and the marginal at horizon n is no longer recoverable from a single fixed step distribution.

## What this experiment shows

The convolution identity *marginal = n-fold convolution of a fixed step* held exactly in E00–E03 and is broken here. The break is quantitatively predictable: the without-replacement marginal sits on a different closed form than the iid one, and the variance shrinks by exactly the finite-population factor (N − n)/(N − 1). The substrate's one-step operator stops being trivial — it now carries the running count of what remains in the urn, and the next step's probability is read off that state.

This is the entry point into the part of the programme where the substrate's dynamics start to matter. From here, the natural next step (E05) is to keep the urn finite but turn the dependence into a memory at a fixed depth — a Markov-like rule that conditions only on the most recent draw rather than on the entire history. That isolates the *memory* effect from the *exhaustion* effect we saw here.
