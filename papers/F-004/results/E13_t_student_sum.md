# E13 — The t_3-sum trajectory  (variance OK, kurtosis infinite)

**Step rule:** Y_t ~ t_3 independent across t.  **Trajectory:** S_t = Y_1 + Y_2 + ... + Y_t.  **Step variance:** ν/(ν − 2) = 3.0.  **Step excess kurtosis:** infinite (the fourth-moment integral diverges for ν ≤ 4).

## What is new in this experiment

Between the all-moments-finite world of E06 – E08 and the no-moments world of E10 there is a wide regime where some moments exist and others do not. Most real-world heavy-tailed processes (financial returns, insurance claims, biological dispersal distances) live in this regime. This experiment covers the simplest case: the t-Student with ν = 3, where the variance is a finite number (3.0) but the fourth moment is infinite.

## Var(S_n) follows the predicted scaling exactly

The variance is a finite-moment quantity. It adds across independent steps and grows linearly with n:

| horizon n | Var(S_n) empirical | predicted n · 3 | median | IQR |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 2.9470 | 3.0 | +0.0003 | 1.5200 |
| 10 | 30.1607 | 30.0 | -0.0071 | 6.3340 |
| 100 | 300.6619 | 300.0 | +0.0211 | 22.2436 |
| 1000 | 3001.2595 | 3000.0 | +0.3459 | 72.6806 |

Empirical Var(S_n) matches the predicted 3n to within sampling noise at every horizon. The variance-based reading works perfectly here, which is the first sign that this regime is *not* the same as E10 (where variance does not exist). It is the *moments above order ν − 1* that fail; the moments up to order 2 are well-defined and behave classically.

## Variance is stable across runs; kurtosis is not

Repeating the experiment five times at each horizon (with 20000 fresh samples each) and reading both variance and kurtosis:

| horizon n | variance across 5 runs | kurtosis across 5 runs |
| ---: | :--- | :--- |
| 1 | [2.9, 2.85, 3.34, 2.95, 2.83] | [17.8, 306.0, 60.4, 29.5, 19.5] |
| 10 | [29.56, 29.36, 31.46, 29.62, 28.59] | [4.6, 12.7, 5.0, 6.2, 583.3] |
| 100 | [302.49, 294.74, 301.09, 311.19, 307.37] | [1.2, 0.9, 2.0, 5.0, 1.2] |
| 1000 | [2945.01, 2922.13, 2992.23, 2946.64, 2971.38] | [0.2, 27.4, 0.4, 0.4, 0.4] |

Variance estimates agree across runs to a few percent — they are consistent estimators of a finite truth. Kurtosis estimates jump around by orders of magnitude with no convergence — there is no finite truth for them to converge to. **This is the empirical statement of "the fourth moment is infinite" as a measurable fact**, parallel to what E10 showed for the second moment of Cauchy.

## The body of the marginal becomes Gaussian; the tail does not

Because the variance is finite, the body of the marginal converges to a Gaussian shape at large n. The tails, however, do not — they carry the heaviness of the step for a long time. We measure both effects.

We measure both effects on a fixed σ-based scale:

  - **body L¹** between the empirical density on |x| ≤ 2σ and the Gaussian of width σ = √(3n) (renormalised to its mass on the body window) — should shrink with n;

  - **tail ratio at k·σ** = P(|S_n| > k·σ) empirical divided by the same probability under N(0, 3n). A ratio above 1 means the empirical tail is heavier than Gaussian at that depth. The ratio should be above 1 and decay slowly with n, more slowly at deeper k.

| horizon n | σ predicted (√3n) | body L¹ | P(|S|>3σ) emp | P(|S|>3σ) gauss | ratio 3σ | P(|S|>5σ) emp | P(|S|>5σ) gauss | ratio 5σ |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 1.7321 | 0.3596 | 0.0140 | 0.0027 | 5.20 | 3.47e-03 | 5.73e-07 | 6043.9 |
| 10 | 5.4772 | 0.1362 | 0.0091 | 0.0027 | 3.37 | 1.37e-03 | 5.73e-07 | 2389.7 |
| 100 | 17.3205 | 0.0460 | 0.0053 | 0.0027 | 1.95 | 5.40e-04 | 5.73e-07 | 941.9 |
| 1000 | 54.7723 | 0.0189 | 0.0038 | 0.0027 | 1.40 | 1.60e-04 | 5.73e-07 | 279.1 |

The body L¹ drops with n: the central region of the marginal fits a Gaussian of width √(3n) increasingly well. The tail ratios stay above 1 — the empirical tail carries excess mass compared to the Gaussian at the same depth — and the ratio at 5σ stays an order of magnitude or more above 1 even at large n. **The marginal is Gaussian in its body and heavier than Gaussian in its extremes**, simultaneously, at every finite horizon. The tail ratio at deeper thresholds is the cleanest diagnostic of the heaviness that variance and L¹-on-body cannot see.

## The substrate's view of this trajectory

The operator is the cumulative identity (independent steps), so all content lives in the step. The step has a *partial* moment structure: mean and variance are finite numbers, third moment is zero by symmetry, fourth moment is infinite. The substrate's reading has to mirror this: use the variance-based summary where it applies (it does, here, for Var(S_n)), and use quantile-based summaries (IQR, body-vs-tail mass ratios) where the moments fail. The mixed reading is the substrate's natural fit to this regime.

## What this experiment shows

Three regimes are now mapped along the tail axis:

  - **All moments finite** (E06 – E08, also E00 – E05, E11 – E12): every moment-based summary works, the decay-of-shape rule from note 02 applies in full;

  - **First few moments finite** (this experiment, t_3): the moments below ν are stable estimators, moments at or above ν fluctuate without converging; the bell still appears in the body of the marginal at large n but the tails stay heavy;

  - **No moments finite** (E10, Cauchy): every moment-based summary is unstable; the marginal stays the same heavy-tailed shape at every horizon, with scale growing linearly in n.

Most heavy-tailed real-world data sits in the middle regime. What this experiment establishes is that the substrate's machinery handles it cleanly by reading what is readable (variance, body shape) and switching to quantile-based diagnostics for what is not (kurtosis, tail mass).
