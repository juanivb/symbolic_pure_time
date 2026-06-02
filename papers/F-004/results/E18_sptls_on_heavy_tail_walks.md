# E18 — SPTLS on three heavy-tailed random walks

**Setup.**  Same dependence structure as E17 (iid steps, the running sum is the trajectory). The step distribution varies across four cases, spanning the tail-thickness axis covered by E13, E14, and E10:

  - **Gaussian** (baseline; same as E17): all moments finite, variance 1;
  - **t-Student ν=3** (E13): finite variance (= 3), infinite kurtosis;
  - **stable α=1.5** (E14): finite mean (= 0), infinite variance; the marginal of S_n is stable with scale n^(2/3);
  - **Cauchy** (E10): no moments; the marginal of S_n is Cauchy with scale n.

All four share the same planted operator on the kinematic jet:

    M̂_planted  =  [[ 1.0000,  0.0000,  0.0000],
                   [ 0.0000,  0.0000,  0.0000],
                   [ 0.0000, -1.0000,  0.0000]].

## Recovery quality vs tail thickness

Frobenius distance ‖M̂_emp − M̂_planted‖_F across 5 independent runs at N = 200000:

| DGP | description | min Frob | median Frob | max Frob | max/min ratio |
| :--- | :--- | ---: | ---: | ---: | ---: |
| Gaussian (baseline, from E17) | all moments finite | 0.0064 | 0.0084 | 0.0090 | 1.41 |
| t-Student ν=3 | variance finite, kurtosis infinite | 0.0013 | 0.0095 | 0.0166 | 12.54 |
| stable α=1.5 | mean finite, variance infinite | 0.0005 | 0.0010 | 0.0030 | 6.26 |
| Cauchy | no moments finite | 0.0001 | 0.0004 | 0.0013 | 15.31 |

## Reading the table

The result is **the opposite of what I predicted before running the experiment.** I expected the recovery to degrade as the tail got heavier — Gaussian best, Cauchy worst. Honest empirical reading: **the recovery gets *better* as the tail gets heavier.** Cauchy median Frobenius is **0.0004**, roughly 20× tighter than the Gaussian baseline's 0.0084. stable α=1.5 sits at 0.0010, t_3 at 0.0095 (a bit worse than Gaussian, presumably because t_3 is heavier-tailed than Gaussian but not heavy enough to trigger the leverage effect described below).

The mechanism that produces this pattern, in plain wording.

The OLS estimator of the unit-root coefficient M̂[0, 0] is

    M̂[0, 0]  =  Σ X_t X_{t+1} / Σ X_t².

For a Gaussian random walk, X_t has standard deviation √t — large but bounded by O(√N). The increment ε_{t+1} has variance 1. So X_{t+1} / X_t = 1 + ε_{t+1}/X_t, and the relative error of M̂[0, 0] is dominated by ε / |X| ~ 1/√t at large t. OLS noise scales as 1/√N.

For a Cauchy random walk, X_t can be **enormously** large at random times (because the cumulative sum of Cauchys has scale t rather than √t, and Cauchy's extreme values are very extreme). The increment ε_{t+1} stays Cauchy with scale 1. On those enormous samples, ε_{t+1}/X_t is tiny — *relative* error vanishes. **The Cauchy walk's extreme values give the OLS regression super-leverage on the unit-root coefficient**: a few samples carry essentially infinite information about M̂[0, 0], because the relationship X_{t+1} = X_t + ε is essentially exact when |X_t| is enormous.

The same mechanism applies to the other planted entries (M̂[2, 1] = −1). The OLS recovers each entry where the algebraic identity holds with vanishing relative error on the dominant samples. **Heavier-tailed levels and increments make those dominant samples more extreme, which makes the operator estimate more precise.**

This is a known phenomenon in the unit-root and integrated-process literature (super-consistency of OLS under infinite variance), but it lands here as a clean empirical fact: heavy-tail walks are an *easier* environment for substrate-operator recovery, not a harder one.

## What this experiment establishes

The substrate operator on the jet is **recoverable across the entire tail-thickness axis**, from Gaussian to Cauchy, with precision that grows rather than shrinks as the tail gets heavier (in the unit-root regime where the level dominates). The operator's structure depends only on the dependence structure of the process (iid here in all four cases), not on the moments of the step.

**The empirical statement of this is the heart of the substrate philosophy for F-004:** the *operator* is what the substrate reads; the *residual* is where moment-based summaries live. When residual moments fail, switch to quantile-based summaries — but the operator itself is robust because of the unit-root super-leverage. The substrate's primary reading is **more** reliable on heavy-tailed random walks than on Gaussian ones.

This finding contradicts the conventional intuition that "heavy tails make estimation harder". It is correct for *stationary* parameter estimation; it is wrong for *unit-root* parameter estimation, and the jet representation of a random walk lives entirely in the unit-root regime.

## What this enables for next experiments

  - **E19** (log-normal): show that the substrate operator on log X_t recovers the additive random-walk structure cleanly, even though log X_t = sum of Gaussians (so the substrate sees exactly the E17 case, just in different coordinates).

  - **E20** (multivariate / coupled trajectories): extend the jet to two coupled processes and read the cross-channel operator components — the multivariate generalisation of E15 / E16.

  - **E21** (real data application): take a financial return series, apply the same jet-SPTLS, read the operator, classify the dependence + tail regime from the substrate fingerprint.
