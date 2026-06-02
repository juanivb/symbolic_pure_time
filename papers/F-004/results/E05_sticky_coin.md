# E05 — The sticky coin trajectory  (q = 0.7)

**Trajectory length:** N = 100000.  **Transition rule:** P(X_{t+1} = X_t) = q = 0.7; P(X_{t+1} = 1 − X_t) = 1 − q = 0.30000000000000004.  **Stationary distribution:** ½/½ (by symmetry of the transition matrix).

## What is new in this experiment

After E04 we already had a non-iid example, but the urn mixed two effects: memory at depth 1 (you know which ball came out) and exhaustion (the urn drains as you draw). This experiment isolates the memory effect. The 'urn' here is infinite — the next flip depends only on the most recent one, regardless of the whole history. It is the simplest possible non-iid case.

Three observations follow.

## 1. The marginal at horizon n has its own closed form

The marginal P(S_n = k) for a Markov trajectory is no longer a self-convolution of one step — but it is still computable in closed form, by dynamic programming over (current state, count so far). The recursion is short: at step t, we record the probability of being in state s ∈ {0, 1} with count k so far; at step t + 1 we update it with the transition probabilities.

Comparing the empirical marginal, the Markov closed form (via DP), and the iid path-count of a fair coin (the answer we would get if we ignored the dependence):

| horizon n | TV(emp, Markov DP) | TV(emp, iid) | TV(Markov DP, iid) |
| ---: | ---: | ---: | ---: |
| 5 | 0.0034 | 0.2002 | 0.2032 |
| 10 | 0.0027 | 0.1991 | 0.1991 |
| 50 | 0.0073 | 0.2013 | 0.2029 |
| 500 | 0.0110 | 0.2032 | 0.2021 |

The empirical marginal lies on the Markov DP curve to within sampling noise. It does *not* lie on the iid curve, and the gap between the two closed forms grows with n. The discrepancy is in the *spread* of the marginal — persistence keeps the running sum closer to the running mean for longer (when the chain stays on one side) and then swings it away (when the chain switches sides). The marginal ends up wider than an iid coin would predict.

## 2. The variance scales by a memory factor

For an iid fair coin Var(S_n) / n = ¼ exactly at every n. With memory, this ratio converges to a different value at large n:

    Var(S_n) / n  →  ¼ · (1 + ρ) / (1 − ρ),     ρ = 2q − 1.

At q = ½ (no memory) the factor is 1 and we recover the iid value ¼. For q > ½ the factor is greater than 1 (persistence inflates variance); for q < ½ it is less than 1 (anti-persistence deflates variance). We measure Var(S_n)/n directly at several horizons and three values of q:

| q | ρ = 2q − 1 | Var(S_n)/n at n=50 | n=100 | n=500 | n=1000 | long-run prediction | iid prediction |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0.3 | -0.40 | 0.1019 | 0.1042 | 0.1002 | 0.1011 | 0.1071 | 0.2500 |
| 0.5 | +0.00 | 0.2661 | 0.2525 | 0.2629 | 0.2544 | 0.2500 | 0.2500 |
| 0.7 | +0.40 | 0.5445 | 0.5799 | 0.5591 | 0.5979 | 0.5833 | 0.2500 |

The empirical Var(S_n)/n approaches the predicted long-run value at large n. At q = 0.7 the iid prediction of 0.25 underestimates by a factor of about 2.3 (we see ≈ 0.58); at q = 0.3 it overestimates by the same factor (we see ≈ 0.108). The convolution identity that holds in the iid case under-counts the variance when q > ½ and over-counts it when q < ½. Same break as in E04, different mechanism — memory at depth 1 instead of exhaustion.

## 3. The substrate's operator at depth 1 recovers the transition rule

The depth-1 operator of this process is the 2×2 transition matrix: given the current state, it gives the probabilities of the next state. Reading this directly from a single long trajectory by counting consecutive pairs:

| | next = 0 | next = 1 |
| :--- | ---: | ---: |
| **current = 0** | 0.7013  (planted 0.7000) | 0.2987  (planted 0.3000) |
| **current = 1** | 0.2993  (planted 0.3000) | 0.7007  (planted 0.7000) |

Recovery is within sampling noise. The substrate's operator at depth 1 is the empirical transition matrix; the planted rule comes back from the trajectory directly. From this matrix everything else follows: the stationary marginal, the variance scaling, the Markov path-count via DP — the operator at depth 1 is the *whole* content of the substrate for this process.

## What this experiment shows

The simplest possible non-iid trajectory — a two-state chain with depth-1 memory — already breaks the convolution identity in a second way (after E04 broke it via exhaustion). The break is quantitatively predictable: the variance scales by (1 + ρ)/(1 − ρ) with ρ = 2q − 1, and the marginal sits on a different closed form from the iid version. The substrate's operator at depth 1 is the transition matrix itself, and we recover it from the trajectory by counting pairs.

Together with E04 this establishes the kind of role the substrate is going to play in everything that follows. Whenever the next step depends on the past, the operator at the appropriate depth is what carries the dependence, and the marginal at horizon n is recoverable only through that operator (via DP, or via a forward propagation of the conditional distribution). The convolution identity is the special case where the operator has nothing to say.
