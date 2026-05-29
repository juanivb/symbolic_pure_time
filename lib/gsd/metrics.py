"""
gsd.metrics — standard time-series performance measures (numpy only).

These are the numbers a practitioner reports in a results table. p-values
for OLS coefficients use a normal (large-sample) approximation so the
library stays dependency-free; with a t-distribution they would differ
only in small samples, and the t-statistics themselves are exact.
"""
from __future__ import annotations
import math
import numpy as np

__all__ = ["rmse", "mae", "mape", "r2", "aic", "bic", "durbin_watson", "norm_sf"]


def _align(pred, targ):
    pred = np.asarray(pred, float).ravel()
    targ = np.asarray(targ, float).ravel()
    return pred, targ


def rmse(pred, targ):
    pred, targ = _align(pred, targ)
    return float(np.sqrt(np.mean((targ - pred) ** 2)))


def mae(pred, targ):
    pred, targ = _align(pred, targ)
    return float(np.mean(np.abs(targ - pred)))


def mape(pred, targ, eps=1e-12):
    pred, targ = _align(pred, targ)
    return float(np.mean(np.abs((targ - pred) / (np.abs(targ) + eps))) * 100.0)


def r2(pred, targ):
    pred, targ = _align(pred, targ)
    rss = np.sum((targ - pred) ** 2)
    tss = np.sum((targ - targ.mean()) ** 2)
    return float(1.0 - rss / tss) if tss > 0 else float("nan")


def aic(rss, n, k):
    """Akaike information criterion for a Gaussian-likelihood LS fit."""
    return float(n * math.log(rss / n) + 2 * k)


def bic(rss, n, k):
    """Bayesian information criterion for a Gaussian-likelihood LS fit."""
    return float(n * math.log(rss / n) + k * math.log(n))


def durbin_watson(resid):
    """Durbin-Watson statistic for residual autocorrelation (~2 = no AR(1))."""
    r = np.asarray(resid, float).ravel()
    d = np.sum(np.diff(r) ** 2) / np.sum(r ** 2)
    return float(d)


def norm_sf(x):
    """Upper-tail standard-normal survival function via erf (no scipy)."""
    return 0.5 * math.erfc(x / math.sqrt(2.0))
