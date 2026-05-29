"""
gsd — Geometric Signal Dynamics for time series (numpy-only build)
==================================================================

A compact, dependency-light library that fits a series two ways and lets
you *read* the result:

    OLS    ordinary least squares on lagged values  — the review baseline,
           univariate AR(p) and multivariate VAR(p), with a statsmodels-style
           regression summary.
    SPTLS  the non-abelian one-step fit on the kinematic embedding
           q(t)=(z, Dz, D2z) — the programme's estimator, read by polar
           decomposition (rotor x stretch) with full-Mahalanobis whitening.

The library's distinctive piece is the white-box diagnostic suite
(`gsd.diagnostics.SPTLSReport`): the SPTLS analogue of ACF/PACF and the
unit-circle root plot. It characterises a process (stationary / unit-root /
explosive, oscillatory or not) and, for several series, reads the directed
cross-effects (which series — or its velocity / acceleration — drives which)
and a cointegration reading from the joint embedding Gram.

Layout
------
    gsd.embedding      kinematic embedding, univariate + stacked
    gsd.algebra        Cl(3,0)/gl(3) primitives: polar, grades, spectrum
    gsd.forecasting    OLS / SPTLS estimators and their result objects
    gsd.diagnostics    SPTLSReport — the interpretation suite (.summary/.plot)
    gsd.metrics        RMSE, MAE, MAPE, R2, AIC, BIC, Durbin-Watson
    gsd.validation     rolling-origin (walk-forward) cross-validation
    gsd.datasets       reproducible synthetic processes

Quick start
-----------
    import numpy as np, gsd
    y = gsd.datasets.rotation()
    print(gsd.OLS(2).fit(y).summary())          # AR(2) regression table
    res = gsd.SPTLS().fit(y)
    print(res.report().summary())               # phase-space diagnostic
    cv = gsd.validation.compare({"AR(2)": gsd.OLS(2), "SPTLS": gsd.SPTLS()}, y)
    print(cv["_table"])                          # walk-forward comparison

Acknowledgment
--------------
The OLS regression summary deliberately follows the layout popularised by
`statsmodels` (Seabold & Perktold, 2010); we reimplement it in plain numpy
so the library carries no heavy dependency.
"""
from __future__ import annotations

from . import algebra, embedding, metrics, forecasting, diagnostics, validation, datasets
from .forecasting import OLS, SPTLS, OLSResult, SPTLSResult, VARResult
from .diagnostics import SPTLSReport
from .embedding import kinematic_embedding, stack_embedding

__version__ = "0.1.0"

__all__ = [
    "OLS", "SPTLS", "OLSResult", "SPTLSResult", "VARResult", "SPTLSReport",
    "kinematic_embedding", "stack_embedding",
    "algebra", "embedding", "metrics", "forecasting", "diagnostics",
    "validation", "datasets", "__version__",
]
