# gsd — Geometric Signal Dynamics (numpy-only build)

A compact time-series library that fits a series two ways and lets you
*read* the result as a white box:

- **OLS** — ordinary least squares on lagged values (the review baseline):
  univariate `AR(p)` and multivariate `VAR(p)`, with a `statsmodels`-style
  regression summary (coef, std err, t, p, CI, R², AIC, BIC, Durbin–Watson).
- **SPTLS** — the non-abelian one-step fit on the kinematic embedding
  `q(t) = (z, Δz, Δ²z)`: the operator `M̂` read by polar decomposition
  (rotor × stretch) with full-Mahalanobis whitening.

The distinctive piece is the **diagnostic / interpretation suite**
(`gsd.diagnostics.SPTLSReport`) — the SPTLS analogue of Box–Jenkins ACF/PACF
and the unit-circle root plot, for *understanding* a process rather than
forecasting it:

- *univariate*: polar readout (rotor angle/axis, stretch gains, antisymmetry
  index), operator spectrum (the SPTLS "roots"), grade energy
  (scalar/stretch/rotation), a descriptive regime reading, and the PGCF
  grade-0 vs grade-2 curves;
- *multivariate*: per-series process nature (read descriptively as stable /
  near-unit / explosive, oscillatory or not), a **directed cross-effect**
  reading (which series — or its velocity/acceleration — moves which, split
  into coaxial and rotational channels), and a **shared-direction** reading of
  the joint embedding Gram (candidate near-stationary combinations of the
  levels).

These are **descriptive readings of the fitted operator, not statistical
tests.** Formal tests built on them (stationarity, shared directions,
directional coupling) are under development and will ship with their null
distributions. A directed causal *graph* over the influence matrix is a
planned next iteration; the influence matrix already carries the same
information.

## Quick start

```python
import gsd
y = gsd.datasets.rotation()
print(gsd.OLS(2).fit(y).summary())        # AR(2) regression table
res = gsd.SPTLS().fit(y)
print(res.report().summary())             # phase-space diagnostic
res.report().plot()                       # visual companion (pedagogical)

cv = gsd.validation.compare(
    {"AR(2)": gsd.OLS(2), "SPTLS": gsd.SPTLS()}, y)
print(cv["_table"])                        # walk-forward (rolling-origin) CV
```

## Modules

| module | role |
|---|---|
| `gsd.embedding` | kinematic embedding, univariate + stacked (multivariate) |
| `gsd.algebra` | Cl(3,0)/gl(3) primitives: polar, grades, spectrum |
| `gsd.forecasting` | `OLS` / `SPTLS` estimators and result objects |
| `gsd.diagnostics` | `SPTLSReport` interpretation suite (`.summary` / `.plot`) |
| `gsd.metrics` | RMSE, MAE, MAPE, R², AIC, BIC, Durbin–Watson |
| `gsd.validation` | rolling-origin (walk-forward) cross-validation |
| `gsd.datasets` | reproducible synthetic processes |

Dependencies: **numpy** for everything; **matplotlib** only inside `.plot()`.

## Status: soft transition

This is the clean, single-package rebuild. It lives under `lib/gsd_ng/`
during the transition so the legacy research library at `lib/gsd/` (which the
F- and A-series replication scripts still import) keeps working untouched. A
snapshot of the legacy library is preserved outside the public repo at
`research_hub/_legacy_backups/`. Once the paper replication scripts are
migrated to this package, the legacy `lib/gsd/` is removed and this package
takes its place at `lib/gsd/`.

To use it now, put `lib/gsd_ng` on the path:

```python
import sys; sys.path.insert(0, ".../symbolic_pure_time/lib/gsd_ng")
import gsd
```

## Acknowledgment

The OLS regression summary follows the layout popularised by `statsmodels`
(Seabold & Perktold, 2010); it is reimplemented here in plain numpy so the
library carries no heavy dependency.

## Nonlinear dynamics: the quadratic dictionary

`gsd.SPTLS()` fits a linear one-step operator (a VAR(2)-class predictor in a
jet basis), so on linear data it ties a well-specified VAR. For **nonlinear**
dynamics use `gsd.SPTLS(dictionary="quadratic")`, which augments the embedding
with the polynomial degree-2 closure of the levels (z_i z_j) and recovers maps
a linear model cannot represent (e.g. the logistic map, or coupled quadratic
systems) — forecasting only.

*Terminology.* `"quadratic"` is the **polynomial degree** (z_i z_j terms), a
different axis from the Clifford **grade**. The grade-2 (bivector / rotational)
content lives in the linear operator's rotor and is read off every linear fit —
it is always present, never opt-in. (`dictionary="grade2"` is kept as a
deprecated alias for `"quadratic"`.)
