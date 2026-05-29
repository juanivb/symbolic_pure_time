# VAR vs SPTLS (multivariate, multi-step)

`var_vs_sptls.ipynb` — SPTLS run out of the box vs the right model in each
regime:

1. **linear (stationary & cointegrated)** — ties a tuned VAR and a Johansen VECM, reproducing the error-correction without estimating a cointegrating rank (no
   model selection; the payoff is convenience + the white-box reading);
2. **nonlinear, one step** — `SPTLS(dictionary="quadratic")` recovers a map a
   linear model cannot represent at any lag (R²≈1 vs ~0.6–0.97);
3. **deterministic chaos** — SPTLS forecasts to the predictability (Lyapunov)
   horizon while a linear VAR fails structurally from step 1; observational
   noise shortens that horizon for every method (chaos, not a model flaw).

Honest framing: SPTLS does not beat a correctly-specified model on linear data;
it wins when the dynamics are nonlinear. Needs only the `gsd` library.
