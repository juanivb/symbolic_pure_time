# gsd library tutorials

Two notebooks, in order:

1. **`gsd_quickstart.ipynb`** — the minimum an econometrician needs to *apply*
   the library: fit `OLS` (AR/VAR) and `SPTLS`, read the regression summary,
   forecast one step, and run walk-forward (rolling-origin) validation —
   univariate and multivariate.

2. **`gsd_diagnostics.ipynb`** — the *interpretation* suite (`SPTLSReport`): the
   SPTLS analogue of ACF/PACF and the unit-circle root plot. Polar readout,
   operator spectrum, grade energy, PGCF curves, and — for several series —
   directed cross-effects and a shared-direction reading. These are
   **descriptive readings of the fitted operator, not statistical tests**;
   formal tests are under development.

Both need only the `gsd` library (numpy; matplotlib for plots). Each notebook's
first cell points Python at `../../lib`.
