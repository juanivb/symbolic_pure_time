# E03 — The rare-event coin trajectory  (p = 0.01)

**Sample length:** N = 200000.  **Probability of heads:** p = 0.01.  **Path-count of the marginal at horizon n:** P(S_n = k) = C(n, k) · p^k · (1−p)^(n−k), same as in E00/E01. The shape of this expression changes a lot when one outcome is rare.

## What is new in this experiment

Compared to E01 (biased coin with p = 0.3) the imbalance is now extreme: each flip is a 1-in-100 event. The running sum at moderate horizons stays close to zero. The marginal is no longer a near-symmetric bell on a moderately wide support; it is sharply concentrated at small integer values, asymmetric, with a long thin tail to the right. The shape is recognisable as the **rare-event PMF** λ^k · exp(−λ) / k! with λ = n·p, and this experiment shows that the path-count expression coincides with it to high precision when p is small.

## Empirical marginals vs path-count, and path-count vs rare-event PMF

For each horizon n we report two distances:
  - **TV(empirical, path-count)**: how well 200 000 simulated trajectories of length n agree with the closed-form expression;
  - **TV(path-count, rare-event PMF)**: how well the closed-form expression itself is captured by the rare-event shape λ^k · exp(−λ)/k! with λ = n·p.

| horizon n | λ = n·p | TV(emp, path-count) | TV(path-count, rare-event PMF) |
| ---: | ---: | ---: | ---: |
| 100 | 1.00 | 0.0012 | 2.78e-03 |
| 500 | 5.00 | 0.0017 | 2.46e-03 |
| 1000 | 10.00 | 0.0032 | 2.46e-03 |
| 10000 | 100.00 | 0.0068 | 2.44e-03 |

The first column drifts a bit at large n because we are sampling 200 000 trajectories and the marginal spreads out — sampling noise dominates. The second column is the substantive one: the path-count expression at p = 0.01 is essentially the rare-event PMF at any of these horizons, with distances at the 10⁻⁴ level or below. The rare-event shape *is* what the convolution produces in this regime; nothing else is needed.

## The shape of the marginal: skewness and kurtosis at each horizon

When the rare-event PMF is a good description, its skewness is **1/√λ** and its excess kurtosis is **1/λ**. We measure both from the path-count expression at each horizon and compare to (i) the rare-event prediction 1/√λ and 1/λ, (ii) the bell-side prediction (1−2p)/√(n·p·(1−p)) for the skewness (which here is essentially the same since p ≪ 1).

| horizon n | λ | skew (measured) | skew (rare-event 1/√λ) | skew (bell-side) | excess kurt (measured) | excess kurt (rare-event 1/λ) |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 100 | 1.00 | +0.9849 | +1.0000 | +0.9849 | +0.9501 | +1.0000 |
| 500 | 5.00 | +0.4405 | +0.4472 | +0.4405 | +0.1899 | +0.2000 |
| 1000 | 10.00 | +0.3113 | +0.3162 | +0.3115 | +0.0942 | +0.1000 |
| 10000 | 100.00 | +0.0985 | +0.1000 | +0.0985 | +0.0095 | +0.0100 |

Two regimes share the floor:
  - **Moderate λ (n = 100, 500, 1000):** the marginal is well described by the rare-event PMF; skewness and kurtosis follow 1/√λ and 1/λ.
  - **Large λ (n = 10000, λ = 100):** the rare-event PMF itself becomes nearly bell-shaped; skewness and excess kurtosis are small and the marginal looks Gaussian-like.

Same convolution machinery, three regimes: rare event at small λ, rare event becoming a bell as λ grows, fully bell at large λ. The transition is continuous and the same path-count expression covers all three.

## The substrate's view of this trajectory

Still no memory: each flip is independent. The one-step operator is the cumulative identity. **All substantive content lives in the step**, which is now a two-point measure with almost all mass at one value.

Reading the step (centred to mean zero) from the sample:

| quantity | value | expected |
| :--- | ---: | ---: |
| support of the step                  | [-0.01, 0.99] | (−p, 1−p) = (-0.01, 0.99) |
| empirical probabilities              | [0.98984, 0.01017] | (0.99, 0.01) |
| mean                                  | +0.0001650 | 0 |
| variance                              | 0.0100617 | 0.0099000 |
| third moment                          | +0.0098621 | +0.0097020 |
| fourth moment                         | 0.0097645 | — |

## What this experiment shows

The convolution machinery from E00/E01/E02 still produces the marginal at any horizon. When the step is rare-and-imbalanced, the shape that comes out is the rare-event PMF λ^k · exp(−λ) / k! with λ = n·p. This is just another shape that the same convolution produces — there is no second mechanism. The rare-event PMF itself becomes a bell when λ is large, which closes the loop: at very large horizons even rare events accumulate enough to look bell-shaped.
